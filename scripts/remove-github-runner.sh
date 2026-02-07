#!/bin/bash
# Remove GitHub Actions self-hosted runner
# Usage: ./scripts/remove-github-runner.sh

set -euo pipefail

RUNNER_DIR="${HOME}/actions-runner"

echo "=== GitHub Actions Runner Removal ==="
echo ""

if [ ! -d "${RUNNER_DIR}" ]; then
    echo "Runner directory not found at ${RUNNER_DIR}"
    echo "Nothing to remove."
    exit 0
fi

cd "${RUNNER_DIR}"

# Stop and uninstall service
if [ -f "svc.sh" ]; then
    echo "Stopping runner service..."
    sudo ./svc.sh stop 2>/dev/null || true
    echo "Uninstalling runner service..."
    sudo ./svc.sh uninstall 2>/dev/null || true
fi

# Unconfigure runner
echo ""
echo "To remove the runner from GitHub, you need a removal token."
echo ""
echo "Get one with:"
echo "  gh api repos/bencappello/health-predict/actions/runners/remove-token \\"
echo "    --method POST --jq '.token'"
echo ""
echo "Then run:"
echo "  cd ${RUNNER_DIR}"
echo "  ./config.sh remove --token <REMOVAL_TOKEN>"
echo ""

if [ -n "${RUNNER_TOKEN:-}" ]; then
    echo "Using RUNNER_TOKEN from environment..."
    ./config.sh remove --token "${RUNNER_TOKEN}" || true
fi

echo ""
echo "To completely remove runner files:"
echo "  rm -rf ${RUNNER_DIR}"
echo ""
echo "=== Removal Complete ==="
