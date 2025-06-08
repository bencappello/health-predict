#!/usr/bin/env python3
"""
Local debugging script for verify_deployment function
This allows us to test the deployment verification logic without a working Kubernetes cluster
"""

import subprocess
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mock_kubectl_commands():
    """Mock kubectl commands to simulate various scenarios"""
    print("=== DEBUGGING KUBECTL CONNECTIVITY ===")
    
    # Test basic kubectl connectivity
    print("\n1. Testing kubectl cluster-info:")
    result = subprocess.run(["kubectl", "cluster-info"], capture_output=True, text=True, check=False)
    print(f"Exit code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    
    print("\n2. Testing kubectl get nodes:")
    result = subprocess.run(["kubectl", "get", "nodes"], capture_output=True, text=True, check=False)
    print(f"Exit code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    
    print("\n3. Testing kubectl get namespaces:")
    result = subprocess.run(["kubectl", "get", "namespaces"], capture_output=True, text=True, check=False)
    print(f"Exit code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

def test_deployment_verification():
    """Test the deployment verification commands used in the DAG"""
    namespace = "health-predict"
    deployment_name = "health-predict-api"
    
    print(f"\n=== TESTING DEPLOYMENT VERIFICATION ===")
    print(f"Namespace: {namespace}")
    print(f"Deployment: {deployment_name}")
    
    # Test 1: Rollout status
    print(f"\n1. Testing rollout status:")
    rollout_result = subprocess.run([
        "kubectl", "rollout", "status", f"deployment/{deployment_name}",
        "-n", namespace, "--timeout=60s"
    ], capture_output=True, text=True, check=False)
    
    print(f"Rollout status - Exit code: {rollout_result.returncode}")
    print(f"Rollout status - Stdout: '{rollout_result.stdout}'")
    print(f"Rollout status - Stderr: '{rollout_result.stderr}'")
    
    # Test 2: Get pods
    print(f"\n2. Testing get pods:")
    pods_result = subprocess.run([
        "kubectl", "get", "pods", "-n", namespace, 
        "-l", "app=health-predict-api", 
        "--field-selector=status.phase=Running",
        "-o", "jsonpath={.items[*].metadata.name}"
    ], capture_output=True, text=True, check=False)
    
    print(f"Pods command - Exit code: {pods_result.returncode}")
    print(f"Pods command - Stdout: '{pods_result.stdout}'")
    print(f"Pods command - Stderr: '{pods_result.stderr}'")
    
    # Test 3: Check for any deployments in the namespace
    print(f"\n3. Testing get deployments:")
    deploy_result = subprocess.run([
        "kubectl", "get", "deployments", "-n", namespace
    ], capture_output=True, text=True, check=False)
    
    print(f"Deployments - Exit code: {deploy_result.returncode}")
    print(f"Deployments - Stdout: '{deploy_result.stdout}'")
    print(f"Deployments - Stderr: '{deploy_result.stderr}'")
    
    # Test 4: Check if namespace exists
    print(f"\n4. Testing namespace existence:")
    ns_result = subprocess.run([
        "kubectl", "get", "namespace", namespace
    ], capture_output=True, text=True, check=False)
    
    print(f"Namespace check - Exit code: {ns_result.returncode}")
    print(f"Namespace check - Stdout: '{ns_result.stdout}'")
    print(f"Namespace check - Stderr: '{ns_result.stderr}'")

def check_env_vars():
    """Check the environment variables that would be used in the DAG"""
    print("\n=== CHECKING ENVIRONMENT VARIABLES ===")
    
    # Load environment variables (same as DAG)
    from dotenv import load_dotenv
    load_dotenv('../.env')
    
    k8s_namespace = os.getenv('K8S_NAMESPACE', 'health-predict')
    k8s_deployment = os.getenv('K8S_DEPLOYMENT_NAME', 'health-predict-api')
    
    print(f"K8S_NAMESPACE: {k8s_namespace}")
    print(f"K8S_DEPLOYMENT_NAME: {k8s_deployment}")

if __name__ == "__main__":
    print("=== VERIFY DEPLOYMENT DEBUGGING ===")
    
    # Check environment variables
    check_env_vars()
    
    # Test kubectl connectivity
    mock_kubectl_commands()
    
    # Test specific deployment verification commands
    test_deployment_verification()
    
    print("\n=== DEBUGGING COMPLETE ===")
    print("Next steps:")
    print("1. If kubectl connectivity fails, restart Minikube")
    print("2. If namespace doesn't exist, create it")
    print("3. If deployment doesn't exist, check the deploy_to_kubernetes step")
    print("4. Update verify_deployment function based on findings") 