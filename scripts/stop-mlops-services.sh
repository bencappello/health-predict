#!/bin/bash

# MLOps Services Stop Script
# This script properly stops all MLOps services including Minikube
# Usage: ./scripts/stop-mlops-services.sh [--keep-minikube] [--remove-volumes]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLOPS_SERVICES_DIR="$PROJECT_ROOT/mlops-services"
ENV_FILE="$PROJECT_ROOT/.env"

# Parse command line arguments
KEEP_MINIKUBE=false
REMOVE_VOLUMES=false
for arg in "$@"; do
    case $arg in
        --keep-minikube)
            KEEP_MINIKUBE=true
            shift
            ;;
        --remove-volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--keep-minikube] [--remove-volumes] [--help]"
            echo "  --keep-minikube:   Don't stop Minikube cluster"
            echo "  --remove-volumes:  Remove Docker volumes (WARNING: deletes data)"
            echo "  --help:           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

stop_docker_services() {
    log_info "Stopping Docker Compose services..."
    
    cd "$MLOPS_SERVICES_DIR"
    
    if [ "$REMOVE_VOLUMES" = true ]; then
        log_warning "Removing Docker volumes (this will delete all data)..."
        docker compose --env-file "$ENV_FILE" down -v
    else
        docker compose --env-file "$ENV_FILE" down
    fi
    
    log_success "Docker services stopped"
}

stop_minikube() {
    if [ "$KEEP_MINIKUBE" = true ]; then
        log_info "Keeping Minikube running as requested"
        return 0
    fi

    log_info "Stopping Minikube..."

    # Use 'minikube delete' instead of 'minikube stop' to fully remove the
    # container. A stopped container retains a reference to its Docker network
    # by internal ID. If that network is later pruned (or removed by any cleanup),
    # the next 'minikube start' fails with "network <id> not found" and requires
    # a full 'minikube delete --all --purge' to recover. Deleting up front avoids
    # this entirely — minikube state (K8s deployments, secrets) is ephemeral and
    # gets recreated by the start script anyway.
    if minikube status >/dev/null 2>&1; then
        minikube delete
        log_success "Minikube deleted (will be recreated on next start)"
    else
        # Even if not running, there may be a stopped container with stale state
        minikube delete >/dev/null 2>&1 || true
        log_info "Minikube is not running (cleaned up any residual state)"
    fi
}

cleanup_networks() {
    # Intentionally NOT running 'docker network prune' here.
    # Pruning removes networks that stopped containers still reference by ID,
    # which causes "network not found" errors on the next start. Since we use
    # 'minikube delete' (not stop) above, there are no orphaned networks to
    # worry about — compose networks are removed by 'docker compose down' and
    # minikube's network is removed by 'minikube delete'.
    log_info "Network cleanup: skipped (handled by compose down + minikube delete)"
}

display_status() {
    log_info "Final Status Check:"
    echo "=================================="
    
    # Check Docker services
    cd "$MLOPS_SERVICES_DIR"
    if docker compose --env-file "$ENV_FILE" ps | grep -q "Up"; then
        log_warning "Some Docker services are still running:"
        docker compose --env-file "$ENV_FILE" ps
    else
        log_success "✓ All Docker services stopped"
    fi
    
    # Check Minikube
    if minikube status >/dev/null 2>&1; then
        if [ "$KEEP_MINIKUBE" = true ]; then
            log_info "✓ Minikube is running (kept as requested)"
        else
            log_warning "⚠ Minikube is still running"
        fi
    else
        log_success "✓ Minikube stopped"
    fi
    
    echo "=================================="
}

main() {
    log_info "Starting MLOps Services Stop Script"
    log_info "Project root: $PROJECT_ROOT"
    
    if [ "$REMOVE_VOLUMES" = true ]; then
        log_warning "WARNING: This will remove all Docker volumes and delete data!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Operation cancelled"
            exit 0
        fi
    fi
    
    echo
    
    # Stop services
    stop_docker_services
    stop_minikube
    cleanup_networks
    
    # Show final status
    echo
    display_status
    
    log_success "MLOps services stop completed!"
    
    if [ "$KEEP_MINIKUBE" = false ] && [ "$REMOVE_VOLUMES" = false ]; then
        echo
        echo "💡 Quick restart: ./scripts/start-mlops-services.sh"
        echo "🔄 Full reset:   ./scripts/start-mlops-services.sh --reset"
    fi
}

# Run main function
main "$@" 