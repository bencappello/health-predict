#!/bin/bash

# rebuild-ec2.sh
# Run this script *inside your EC2 instance* to restore the full working environment after terraform apply.

# === CONFIGURATION ===
REPO_SSH_URL="git@github.com:your-username/health-predict-mlops.git"
REPO_DIR="health-predict-mlops"
COMPOSE_DIR="mlops-services"  # This is where docker-compose.yml should live

# === 1. Setup Git (if needed) ===
echo "📦 Ensuring Git is installed..."
sudo apt-get update -y
sudo apt-get install -y git

# === 2. Clone or Pull GitHub Repo ===
if [ ! -d "$REPO_DIR" ]; then
  echo "📥 Cloning repository from $REPO_SSH_URL..."
  git clone "$REPO_SSH_URL"
else
  echo "🔄 Repo already exists. Pulling latest changes..."
  cd "$REPO_DIR" && git pull && cd ..
fi

# === 3. Restore Docker Services ===
echo "🐳 Setting up Docker services..."
mkdir -p ~/"$COMPOSE_DIR"
cp "$REPO_DIR/docker-compose.yml" ~/"$COMPOSE_DIR"/docker-compose.yml

cd ~/"$COMPOSE_DIR"
docker-compose up -d

# === 4. Check Services ===
echo "✅ Docker containers running:"
docker ps

# === 5. Optional: Verify UIs ===
echo "🌐 Airflow UI: http://<EC2_PUBLIC_IP>:8080"
echo "🌐 MLflow UI: http://<EC2_PUBLIC_IP>:5000"

echo "🧹 Rebuild complete."
