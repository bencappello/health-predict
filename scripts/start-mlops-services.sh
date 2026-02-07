#!/bin/bash

# MLOps Services Startup Script
# This script ensures all MLOps services are properly started and healthy
# Usage: ./scripts/start-mlops-services.sh [--rebuild] [--reset]

set -e  # Exit on any error

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
MAX_RETRIES=30
RETRY_DELAY=10

# Parse command line arguments
REBUILD=false
RESET=false
for arg in "$@"; do
    case $arg in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --reset)
            RESET=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--rebuild] [--reset] [--help]"
            echo "  --rebuild: Rebuild Docker images"
            echo "  --reset:   Reset all services (removes volumes)"
            echo "  --help:    Show this help message"
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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for a condition with timeout
wait_for_condition() {
    local condition_func=$1
    local description=$2
    local max_attempts=${3:-$MAX_RETRIES}
    local delay=${4:-$RETRY_DELAY}
    
    log_info "Waiting for: $description"
    
    for i in $(seq 1 $max_attempts); do
        if $condition_func; then
            log_success "$description - Ready!"
            return 0
        fi
        
        log_info "Attempt $i/$max_attempts - $description not ready, waiting ${delay}s..."
        sleep $delay
    done
    
    log_error "$description failed after $max_attempts attempts"
    return 1
}

# Health check functions
check_docker() {
    docker info >/dev/null 2>&1
}

check_minikube_running() {
    minikube status >/dev/null 2>&1 && \
    kubectl cluster-info >/dev/null 2>&1
}

check_ecr_secret() {
    kubectl get secret ecr-registry-key >/dev/null 2>&1
}

create_ecr_secret() {
    log_info "Creating ECR authentication secret..."
    
    # Check if AWS CLI is available and configured
    if ! command_exists "aws"; then
        log_error "AWS CLI not found. Cannot create ECR secret."
        return 1
    fi
    
    # Test AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS credentials not configured. Cannot create ECR secret."
        return 1
    fi
    
    # Get AWS account ID and region from environment or AWS
    local aws_account_id
    local aws_region
    
    # Try to get from environment first
    if [ -f "$ENV_FILE" ]; then
        aws_account_id=$(grep "^AWS_ACCOUNT_ID=" "$ENV_FILE" | cut -d'=' -f2)
        aws_region=$(grep "^AWS_DEFAULT_REGION=" "$ENV_FILE" | cut -d'=' -f2)
    fi
    
    # Fall back to AWS CLI if not in env
    if [ -z "$aws_account_id" ]; then
        aws_account_id=$(aws sts get-caller-identity --query Account --output text)
    fi
    
    if [ -z "$aws_region" ]; then
        aws_region=$(aws configure get region || echo "us-east-1")
    fi
    
    if [ -z "$aws_account_id" ] || [ -z "$aws_region" ]; then
        log_error "Could not determine AWS account ID or region"
        return 1
    fi
    
    local ecr_server="${aws_account_id}.dkr.ecr.${aws_region}.amazonaws.com"
    
    # Create the secret
    if kubectl create secret docker-registry ecr-registry-key \
        --docker-server="$ecr_server" \
        --docker-username=AWS \
        --docker-password="$(aws ecr get-login-password --region "$aws_region")" \
        --namespace=default >/dev/null 2>&1; then
        log_success "ECR authentication secret created successfully"
        return 0
    else
        log_error "Failed to create ECR authentication secret"
        return 1
    fi
}

check_postgres_ready() {
    docker compose -f "$MLOPS_SERVICES_DIR/docker-compose.yml" --env-file "$ENV_FILE" \
        exec -T postgres pg_isready -U airflow >/dev/null 2>&1
}

check_mlflow_ready() {
    curl -f -s http://localhost:5000/health >/dev/null 2>&1 || \
    curl -f -s http://localhost:5000/ >/dev/null 2>&1
}

check_airflow_ready() {
    curl -f -s http://localhost:8080/health >/dev/null 2>&1 || \
    docker compose -f "$MLOPS_SERVICES_DIR/docker-compose.yml" --env-file "$ENV_FILE" \
        exec -T airflow-scheduler airflow dags list >/dev/null 2>&1
}

check_k8s_api_deployment() {
    kubectl get deployment health-predict-api-deployment >/dev/null 2>&1 && \
    kubectl rollout status deployment/health-predict-api-deployment --timeout=60s >/dev/null 2>&1
}

# Main functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local missing_commands=()
    for cmd in docker kubectl minikube curl; do
        if ! command_exists "$cmd"; then
            missing_commands+=("$cmd")
        fi
    done
    
    # Check for docker compose (either docker-compose or docker compose)
    if ! command_exists "docker-compose" && ! docker compose version >/dev/null 2>&1; then
        missing_commands+=("docker-compose")
    fi
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        log_error "Missing required commands: ${missing_commands[*]}"
        log_error "Please install missing dependencies and try again"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file not found: $ENV_FILE"
        log_error "Please create the .env file with required variables"
        exit 1
    fi
    
    # Check Docker is running
    if ! check_docker; then
        log_error "Docker is not running. Please start Docker and try again"
        exit 1
    fi
    
    log_success "All prerequisites checked"
}

start_minikube() {
    log_info "Starting Minikube..."
    
    # Check if Minikube is already running
    if check_minikube_running; then
        log_success "Minikube is already running"
        return 0
    fi
    
    # Clean up any stale Minikube networks that might conflict
    log_info "Cleaning up potential network conflicts..."
    docker network ls --format "{{.Name}}" | grep -E "^minikube$" | xargs -r docker network rm || true
    
    # Start Minikube
    log_info "Starting Minikube cluster..."
    minikube start --driver=docker --cpus=2 --memory=3900MB
    
    # Wait for Minikube to be ready
    if ! wait_for_condition check_minikube_running "Minikube cluster" 15 10; then
        log_error "Failed to start Minikube"
        log_info "Attempting full purge and recreate of Minikube..."
        minikube delete --all --purge || true
        minikube start --driver=docker --cpus=2 --memory=3900MB --force

        if ! wait_for_condition check_minikube_running "Minikube cluster (after purge)" 15 10; then
            log_error "Failed to start Minikube even after purge"
            exit 1
        fi
    fi
    
    # Ensure ECR authentication secret exists (must precede K8s manifests
    # so the pod can pull its image immediately on creation)
    log_info "Checking ECR authentication setup..."
    if check_ecr_secret; then
        log_success "✓ ECR authentication secret already exists"
    else
        log_warning "ECR authentication secret not found, creating it..."
        if create_ecr_secret; then
            log_success "✓ ECR authentication secret created"
        else
            log_error "⚠ Failed to create ECR authentication secret"
            log_warning "ECR image pulls may fail. You may need to create the secret manually:"
            log_warning "kubectl create secret docker-registry ecr-registry-key --docker-server=<ECR_SERVER> --docker-username=AWS --docker-password=\$(aws ecr get-login-password --region <REGION>)"
        fi
    fi

    # Apply Kubernetes manifests if they exist
    if [ -f "$PROJECT_ROOT/k8s/deployment.yaml" ]; then
        log_info "Applying Kubernetes manifests..."
        kubectl apply -f "$PROJECT_ROOT/k8s/"
    fi
    
    log_success "Minikube is ready"
}

start_docker_services() {
    log_info "Starting Docker Compose services..."
    
    cd "$MLOPS_SERVICES_DIR"
    
    # Stop services if reset requested
    if [ "$RESET" = true ]; then
        log_warning "Resetting all services (removing volumes)..."
        docker compose --env-file "$ENV_FILE" down -v
    elif [ "$REBUILD" = true ]; then
        log_info "Rebuilding services..."
        docker compose --env-file "$ENV_FILE" down
    fi
    
    # Start services
    if [ "$REBUILD" = true ]; then
        docker compose --env-file "$ENV_FILE" up -d --build
    else
        docker compose --env-file "$ENV_FILE" up -d
    fi
    
    # Wait for services to be ready
    wait_for_condition check_postgres_ready "PostgreSQL"
    wait_for_condition check_mlflow_ready "MLflow"
    wait_for_condition check_airflow_ready "Airflow"
    
    log_success "All Docker services are ready"
}

verify_system_health() {
    log_info "Performing comprehensive system health check..."
    
    # Check all services
    local services=(
        "check_docker:Docker daemon"
        "check_minikube_running:Minikube cluster" 
        "check_postgres_ready:PostgreSQL"
        "check_mlflow_ready:MLflow"
        "check_airflow_ready:Airflow"
        "check_ecr_secret:ECR authentication"
    )
    
    local failed_services=()
    
    for service in "${services[@]}"; do
        IFS=':' read -r check_func description <<< "$service"
        if $check_func; then
            log_success "✓ $description"
        else
            log_error "✗ $description"
            failed_services+=("$description")
        fi
    done
    
    if [ ${#failed_services[@]} -ne 0 ]; then
        log_error "Health check failed for: ${failed_services[*]}"
        return 1
    fi
    
    # Try to check DAGs
    log_info "Checking Airflow DAGs..."
    if docker compose --env-file "$ENV_FILE" exec -T airflow-scheduler airflow dags list >/dev/null 2>&1; then
        log_success "✓ Airflow DAGs are accessible"
    else
        log_warning "⚠ Airflow DAGs not yet accessible (may need more time)"
    fi
    
    log_success "System health check completed successfully"
    return 0
}

display_service_info() {
    log_info "MLOps Services Information:"
    echo "=================================="
    echo "📊 Airflow UI:    http://localhost:8080 (admin/admin)"
    echo "🧪 MLflow UI:     http://localhost:5000"
    echo "📓 JupyterLab:    http://localhost:8888"
    echo "🗄️  PostgreSQL:    localhost:5432"
    echo "☸️  Kubernetes:   $(minikube ip):$(kubectl get service health-predict-api-service -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo 'N/A')"
    echo "=================================="
    echo
    echo "🔧 Useful commands:"
    echo "  Check status:     docker compose --env-file $ENV_FILE ps"
    echo "  View logs:        docker compose --env-file $ENV_FILE logs -f"
    echo "  Stop services:    docker compose --env-file $ENV_FILE down"
    echo "  Minikube status:  minikube status"
    echo "  Kubectl check:    kubectl get pods"
    echo
}

# Main execution
main() {
    log_info "Starting MLOps Services Startup Script"
    log_info "Project root: $PROJECT_ROOT"
    echo
    
    # Run startup sequence
    check_prerequisites
    start_minikube
    start_docker_services
    
    # Health verification
    log_info "Waiting for all services to stabilize..."
    sleep 15
    
    if verify_system_health; then
        log_success "🎉 All MLOps services are running successfully!"
        echo
        display_service_info
        
        # Final recommendations
        echo "🚀 Next steps:"
        echo "  1. Access Airflow UI and trigger 'health_predict_training_hpo' DAG"
        echo "  2. Once training completes, trigger 'health_predict_continuous_improvement' DAG"
        echo "  3. Monitor services with the provided URLs above"
        echo
        log_success "MLOps environment is ready for use!"
    else
        log_error "Some services are not healthy. Check the logs and try again."
        exit 1
    fi
}

# Run main function
main "$@" 