#!/bin/bash
# EC2 IP Update Helper Script
# Usage: ./scripts/update_ec2_ip.sh <NEW_IP_ADDRESS>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <NEW_IP_ADDRESS>"
    echo "Example: $0 13.222.206.225"
    exit 1
fi

NEW_IP="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

echo "Updating EC2 IP address to: $NEW_IP"

# Update .env file
if grep -q "^EC2_PUBLIC_IP=" "$ENV_FILE"; then
    sed -i "s/^EC2_PUBLIC_IP=.*/EC2_PUBLIC_IP=$NEW_IP/" "$ENV_FILE"
    echo "✓ Updated EC2_PUBLIC_IP in .env"
else
    echo -e "\n# EC2 Instance IP Address (update when instance restarts)\nEC2_PUBLIC_IP=$NEW_IP" >> "$ENV_FILE"
    echo "✓ Added EC2_PUBLIC_IP to .env"
fi

echo ""
echo "EC2 IP updated successfully!"
echo ""
echo "You can now access services at:"
echo "  - Airflow UI: http://$NEW_IP:8080"
echo "  - MLflow UI:  http://$NEW_IP:5000"
echo "  - Jupyter:    http://$NEW_IP:8888"
echo ""
echo "Note: Services are accessible via 'localhost' for ssh tunnels or when on the EC2 instance."
