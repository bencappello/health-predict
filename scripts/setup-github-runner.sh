#!/bin/bash
# Setup GitHub Actions self-hosted runner on EC2
# Usage: ./scripts/setup-github-runner.sh

set -euo pipefail

RUNNER_DIR="${HOME}/actions-runner"
RUNNER_VERSION="2.321.0"
RUNNER_ARCH="linux-x64"
RUNNER_TAR="actions-runner-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_TAR}"
REPO_URL="https://github.com/bencappello/health-predict"

echo "=== GitHub Actions Self-Hosted Runner Setup ==="
echo ""

# Step 1: Download runner
if [ -d "${RUNNER_DIR}" ] && [ -f "${RUNNER_DIR}/run.sh" ]; then
    echo "Runner already installed at ${RUNNER_DIR}"
    echo "To reinstall, remove the directory first: rm -rf ${RUNNER_DIR}"
else
    echo "Step 1: Downloading runner v${RUNNER_VERSION}..."
    mkdir -p "${RUNNER_DIR}"
    cd "${RUNNER_DIR}"
    curl -sL "${RUNNER_URL}" -o "${RUNNER_TAR}"
    tar xzf "${RUNNER_TAR}"
    rm -f "${RUNNER_TAR}"
    echo "Runner downloaded to ${RUNNER_DIR}"
fi

cd "${RUNNER_DIR}"

# Step 2: Guide registration
echo ""
echo "Step 2: Register the runner"
echo "==============================="
echo ""
echo "You need a registration token from GitHub. To get one:"
echo ""
echo "  1. Go to: ${REPO_URL}/settings/actions/runners/new"
echo "  2. Copy the token from the configuration section"
echo ""
echo "  Or use the GitHub CLI:"
echo "    gh api repos/bencappello/health-predict/actions/runners/registration-token \\"
echo "      --method POST --jq '.token'"
echo ""

if [ -n "${RUNNER_TOKEN:-}" ]; then
    echo "Using RUNNER_TOKEN from environment..."
    ./config.sh --url "${REPO_URL}" --token "${RUNNER_TOKEN}" \
        --name "ec2-mlops-runner" \
        --labels "self-hosted,linux,x64,ec2,mlops" \
        --work "_work" \
        --unattended --replace
else
    echo "Run the following command to configure the runner:"
    echo ""
    echo "  cd ${RUNNER_DIR}"
    echo "  ./config.sh --url ${REPO_URL} --token <YOUR_TOKEN> \\"
    echo "    --name ec2-mlops-runner \\"
    echo "    --labels self-hosted,linux,x64,ec2,mlops \\"
    echo "    --work _work --unattended --replace"
    echo ""
fi

# Step 3: Install as service
echo ""
echo "Step 3: Install and start as a systemd service"
echo "================================================"
echo ""
echo "After configuring, install and start the service:"
echo ""
echo "  cd ${RUNNER_DIR}"
echo "  sudo ./svc.sh install"
echo "  sudo ./svc.sh start"
echo "  sudo ./svc.sh status"
echo ""
echo "The runner will start automatically on boot."
echo ""
echo "=== Setup Complete ==="
