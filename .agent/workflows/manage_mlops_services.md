---
description: Complete MLOps stack startup, deployment, Kubernetes, Minikube, Airflow, MLflow services - production environment setup and troubleshooting
---

# MLOps Services Management Workflow

**USE THIS WORKFLOW WHEN:** Starting services, deploying models, launching Kubernetes, setting up production environment, restarting services after EC2 reboot, or troubleshooting MLOps infrastructure.

This workflow provides comprehensive instructions for managing the complete MLOps stack including Kubernetes/Minikube for production model deployment.

---

## 🚀 FULL PRODUCTION SETUP (Default - Use This)

Use this for **complete MLOps pipeline** including automated model deployment via Kubernetes. Required for drift monitoring → retraining → deployment automation.

### Step 1: Use the Automated Startup Script

The project includes a comprehensive startup script that handles everything automatically.

// turbo
```bash
bash scripts/start-mlops-services.sh
```

**What this does:**
- ✅ Validates prerequisites (Docker, kubectl, minikube)
- ✅ Starts Minikube cluster with Kubernetes
- ✅ Applies Kubernetes manifests (`k8s/deployment.yaml`)
- ✅ Creates ECR authentication secrets for pulling images
- ✅ Starts all Docker services (Postgres, MLflow, Airflow, JupyterLab)
- ✅ Performs comprehensive health checks
- ✅ Displays service URLs and next steps

**Expected Output:**
```
🎉 All MLOps services are running successfully!

📊 Airflow UI:    http://localhost:8080 (admin/admin)
🧪 MLflow UI:     http://localhost:5000
📓 JupyterLab:    http://localhost:8888
☸️  Kubernetes:   <minikube-ip>:<node-port>
```

### Step 2: Verify Services Are Running

// turbo
```bash
# Check all containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check Kubernetes
kubectl get pods
kubectl get deployments
kubectl get services
```

### Step 3: Verify Endpoints

// turbo
```bash
# Test Airflow (returns 302 redirect - this is correct)
curl -I http://localhost:8080

# Test MLflow (returns 200 OK)
curl -I http://localhost:5000

# Test Kubernetes API deployment
kubectl get deployment health-predict-api-deployment
```

### Optional: Rebuild Services

If you need to rebuild Docker images (e.g., after code changes):

```bash
bash scripts/start-mlops-services.sh --rebuild
```

### Optional: Reset Everything (Clean Slate)

⚠️ **WARNING:** This removes all volumes and data!

```bash
bash scripts/start-mlops-services.sh --reset
```

---

## 🔧 LIGHTWEIGHT MODE (Troubleshooting Only)

**⚠️ USE ONLY FOR:** Quick troubleshooting, testing Airflow DAGs without deployment, or when Kubernetes is temporarily unavailable.

**❌ DO NOT USE FOR:** Production deployment, drift monitoring with automated deployment, or complete MLOps pipeline demonstration.

This mode **excludes Kubernetes/Minikube** and only starts core services. You will **NOT** be able to deploy models automatically.

### Step 1: Verify Environment

// turbo
```bash
# Check .env exists
if [ ! -f .env ]; then
    echo "❌ CRITICAL: .env file missing"
    exit 1
fi

# Copy to mlops-services if needed
if [ ! -f mlops-services/.env ]; then
    cp .env mlops-services/.env
fi
```

### Step 2: Start Core Services Only

```bash
cd mlops-services
export $(cat ../.env | xargs)
docker compose up -d postgres mlflow airflow-init airflow-scheduler airflow-webserver jupyterlab
```

### Step 3: Verify Health

// turbo
```bash
echo "Checking services..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(postgres|airflow|mlflow)"

# Test endpoints
curl -I http://localhost:5000  # MLflow
curl -I http://localhost:8080  # Airflow
```

---

## 🛠️ Common Troubleshooting

### Issue: Minikube fails to start with network errors

**Symptoms:**
```
Error: network <id> not found
failed to set up container networking
```

**Solution:**
```bash
# Delete and recreate Minikube
minikube delete
bash scripts/start-mlops-services.sh
```

### Issue: Airflow webserver exits immediately

**Check:**
```bash
docker logs mlops-services-airflow-webserver-1 --tail 50
```

**Common Causes:**
- Empty `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` in .env
- Postgres not ready yet
- Airflow DB not initialized

**Solution:**
```bash
# Ensure .env is loaded and restart
cd mlops-services
docker compose restart airflow-init airflow-webserver airflow-scheduler
```

### Issue: "Connection Refused" on localhost

**Check:**
```bash
docker ps  # Container running?
docker logs <container-name>  # Check for errors
netstat -tlnp | grep -E "5000|8080"  # Port actually bound?
```

**Solution:**
- If container is dead, check logs and restart
- Verify port mappings in `docker-compose.yml`
- Check firewall/security groups for EC2

### Issue: Kubernetes deployment fails

**Check:**
```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

**Common Causes:**
- ECR authentication secret missing
- Image pull errors
- Insufficient resources

**Solution:**
```bash
# Recreate ECR secret
kubectl delete secret ecr-registry-key
# Then restart services with: bash scripts/start-mlops-services.sh
```

### Issue: MLflow can't access S3 artifacts

**Check:** AWS credentials in `.env`:
```bash
grep -E "AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY" .env
```

**Solution:**
```bash
# Update .env with valid AWS credentials
# Restart MLflow
cd mlops-services
docker compose restart mlflow
```

---

## 📝 Service URLs Reference

After successful startup, services are available at:

- **Airflow UI**: http://localhost:8080 or http://<EC2_IP>:8080 (admin/admin)
- **MLflow UI**: http://localhost:5000 or http://<EC2_IP>:5000
- **JupyterLab**: http://localhost:8888 or http://<EC2_IP>:8888
- **Kubernetes API**: Via kubectl or at Minikube IP

---

## 🔄 Updating EC2 IP Address

When EC2 instance restarts with a new IP, use the helper script:

// turbo
```bash
./scripts/update_ec2_ip.sh <NEW_IP_ADDRESS>
```

This updates the `EC2_PUBLIC_IP` variable in `.env` for easy reference.

---

## 🎯 Next Steps After Startup

1. **Access Airflow UI** and verify DAGs are loaded
2. **Check MLflow** to ensure experiments are accessible
3. **For deployment testing**: Verify Kubernetes pods are running
4. **For drift monitoring**: All services including Kubernetes must be running

---

## 📋 Quick Reference Commands

```bash
# Start everything (RECOMMENDED)
bash scripts/start-mlops-services.sh

# Stop all services
cd mlops-services && docker compose down

# Stop and remove volumes (clean slate)
cd mlops-services && docker compose down -v

# View service logs
docker compose logs -f airflow-scheduler
docker compose logs -f mlflow

# Check Kubernetes
minikube status
kubectl get all

# Update EC2 IP
./scripts/update_ec2_ip.sh <NEW_IP>
```
