### Key Points
- It seems likely that the networking issue in your MLOps project stems from how tests connect to the API in Minikube, with "Connection Refused" errors from Airflow's LocalExecutor subprocesses.
- Research suggests modifying tests to use the service's ClusterIP instead of NodeIP and NodePort may resolve this, as it leverages the shared Minikube network.
- The evidence leans toward a configuration difference between subprocesses and direct container commands, like `curl`, causing the issue.

---

### Direct Answer

#### Overview
You're experiencing networking issues in your MLOps project, particularly when running API tests in the final deployment step. The tests fail with "Connection Refused" errors, despite the deployment pipeline working otherwise, and `curl` commands from the same container succeed. This suggests a problem with how Airflow's LocalExecutor runs subprocesses for tests.

#### Likely Cause
The issue likely arises because the tests, run as Python subprocesses within Airflow, struggle to connect to the Minikube NodePort service, while direct commands like `curl` work. This could be due to how Minikube's Docker driver on Linux handles network access for subprocesses, potentially isolating them from the host network in ways that affect socket connections.

#### Suggested Solution
To address this, try modifying your test script to use the service's **ClusterIP** instead of the NodeIP and NodePort. Since your Airflow container is connected to the Minikube network, it should access the service directly at `<ClusterIP>:80`. Here's how:

1. **Get the ClusterIP**:
   - Run `kubectl get svc health-predict-api-service -o jsonpath='{.spec.clusterIP}'` to find the IP, e.g., `10.96.0.100`.

2. **Update the Test Script**:
   - In `tests/api/test_api_endpoints.py`, set `API_BASE_URL = f"[invalid url, do not cite]"` (replace with the actual ClusterIP).

3. **Adjust the DAG**:
   - Modify `deployment_pipeline_dag.py`'s `construct_test_command` to pass the ClusterIP, e.g., `MINIKUBE_IP='10.96.0.100' K8S_NODE_PORT='80'`.

This approach should allow tests to connect successfully, as it uses internal cluster networking, avoiding potential NodePort access issues.

#### Additional Checks
- Ensure the Airflow container can ping the ClusterIP: `docker compose exec airflow-scheduler ping 10.96.0.100`.
- Check the API service logs for connection attempts: `kubectl logs -f deployment/health-predict-api-deployment`.
- If issues persist, verify no network policies or firewall rules in Minikube are blocking connections.

This solution is based on research into Minikube and Docker networking, suggesting ClusterIP access is more reliable for internal connections [Minikube Accessing Apps Documentation](https://minikube.sigs.k8s.io/docs/handbook/accessing/) [Kubernetes Service Documentation](https://kubernetes.io/docs/concepts/services-networking/service/).

---

### Survey Note: Detailed Analysis of Networking Issues in MLOps Project

This section provides a comprehensive analysis of the networking issues encountered in your MLOps project, focusing on the deployment pipeline's final step where API tests fail with "Connection Refused" errors. The analysis is grounded in the provided system overview, project work journal, and relevant documentation, aiming to uncover potential causes and propose solutions.

#### System Context and Problem Description

Your "Health Predict" MLOps project aims to predict patient readmission risk using a scalable system leveraging AWS, Docker, Kubernetes (Minikube), Apache Airflow, MLflow, and FastAPI. The deployment pipeline involves training models, committing them to a registry, and deploying an API via a Kubernetes cluster on Minikube, running on an EC2 instance. The pipeline works until the final step, where API tests fail due to networking issues, specifically "Connection Refused" errors when connecting to the API at `[invalid url, do not cite]`.

Key components include:
- **AWS Services**: EC2 hosts Minikube and MLOps services, S3 for data storage, and ECR for Docker images.
- **Docker and Kubernetes**: Minikube uses the Docker driver, with Airflow services connected to both `mlops_network` and `minikube` networks via `docker-compose.yml`.
- **Airflow**: Uses LocalExecutor, orchestrating workflows including the `health_predict_api_deployment` DAG, which includes a `run_api_tests` task.
- **API Deployment**: The API is deployed to Minikube, exposed via a NodePort service (`health-predict-api-service`), with tests run from within the Airflow worker container.

The journal entries from May 15, 2025, detail extensive debugging, revealing that while `curl` commands from the `airflow-scheduler` container succeed, tests run via the PythonOperator in the DAG fail, even at the socket level (direct `socket.connect_ex` fails with error 111, "Connection Refused"). This discrepancy suggests a network access issue specific to subprocesses spawned by Airflow's LocalExecutor.

#### Analysis of Potential Causes

To understand the issue, we analyzed the network configuration and execution environment:

1. **Network Configuration**:
   - Minikube, using the Docker driver on Linux, creates a network (`minikube`) where the cluster node (e.g., IP 192.168.49.2) hosts services. Your Airflow containers are connected to this network, as seen in `docker-compose.yml`:
     ```
     networks:
       - mlops_network
       - minikube
     ```
   - NodePort services should be accessible at `<NodeIP>:<NodePort>` from within the same network, and documentation suggests no tunnel is needed for Linux with the Docker driver [Minikube Accessing Apps Documentation](https://minikube.sigs.k8s.io/docs/handbook/accessing/). However, the failure suggests a limitation in accessing NodePort from subprocesses.

2. **Execution Environment**:
   - Airflow's LocalExecutor runs tasks as subprocesses using `multiprocessing.Process`, which should share the container's network namespace. The `run_api_tests` task uses `PythonOperator`, calling `run_api_tests_callable`, which spawns a subprocess via `subprocess.run(["python", "-m", "pytest", ...])`.
   - Despite this, direct socket tests within `run_api_tests_callable` fail, while `curl` from `docker compose exec airflow-scheduler` succeeds, indicating a discrepancy between how these processes access the network.

3. **Comparison with `curl`**:
   - `curl` working from `docker compose exec` suggests the container's network is functional for direct commands. However, the Python subprocess (via `subprocess.run`) fails, even with direct socket tests, pointing to a potential isolation or configuration difference in how Airflow's LocalExecutor handles subprocesses compared to `docker compose exec`.

4. **Hypothesized Causes**:
   - **Subprocess Isolation**: There might be subtle differences in how `multiprocessing.Process` or `subprocess.run` interacts with the network, possibly due to environment variables, user context, or security settings, though all processes should share the same network namespace in Docker.
   - **Minikube Docker Driver Limitation**: On Linux with the Docker driver, while no tunnel is needed, there could be issues accessing NodePort services from certain container processes, especially subprocesses, due to how Docker networking is implemented.
   - **Timing or Readiness**: Although the DAG includes a `verify_deployment_rollout` task, there might be a race condition where the service isn't fully ready for connections from subprocesses, though "Connection Refused" suggests an immediate block, not a timeout.

#### Proposed Solution and Implementation

Given the analysis, the recommended solution is to modify the test script to use the service's **ClusterIP** instead of NodeIP and NodePort, leveraging internal cluster networking for better reliability. This approach is based on Kubernetes service access patterns, where ClusterIP is accessible from within the cluster's network.

##### Steps to Implement

1. **Retrieve the ClusterIP**:
   - Execute the following command to get the ClusterIP:
     ```bash
     kubectl get svc health-predict-api-service -o jsonpath='{.spec.clusterIP}'
     ```
   - Example output: `10.96.0.100`. This IP is internal to the Kubernetes cluster and should be reachable from containers on the `minikube` network.

2. **Modify the Test Script**:
   - Update `tests/api/test_api_endpoints.py` to use the ClusterIP and service port (80, as per your service definition):
     ```python
     cluster_ip = "10.96.0.100"  # Replace with actual ClusterIP
     API_BASE_URL = f"[invalid url, do not cite]"
     ```
   - Ensure all test functions (`test_health_check`, `test_predict_valid_input`, etc.) use this URL for requests.

3. **Update the DAG**:
   - Modify `deployment_pipeline_dag.py`'s `construct_test_command` to fetch and pass the ClusterIP:
     ```python
     def construct_test_command(**kwargs):
         ti = kwargs["ti"]
         cluster_ip = subprocess.check_output(["kubectl", "get", "svc", "health-predict-api-service", "-o", "jsonpath='{.spec.clusterIP}'"]).decode("utf-8").strip()
         test_command = f"MINIKUBE_IP='{cluster_ip}' K8S_NODE_PORT='80' python -m pytest -v --tb=long --show-capture=no /home/ubuntu/health-predict/tests/api/test_api_endpoints.py"
         return {"test_command": test_command}
     ```
   - This ensures the test script receives the correct IP for internal cluster access.

4. **Execute and Verify**:
   - Trigger the `health_predict_api_deployment` DAG and monitor the `run_api_tests` task. The tests should now connect using the ClusterIP, avoiding NodePort-related issues.

##### Rationale
- Using ClusterIP (`<ClusterIP>:80`) leverages Kubernetes' internal service discovery, which is designed for intra-cluster communication. Since the Airflow container is on the `minikube` network, it should access the service directly, bypassing potential NodePort access restrictions.
- This approach aligns with best practices for accessing services within a Kubernetes cluster, as documented in [Kubernetes Service Documentation](https://kubernetes.io/docs/concepts/services-networking/service/).

#### Additional Troubleshooting and Considerations

If the above solution does not resolve the issue, consider the following steps to further diagnose:

- **Verify Network Connectivity**:
  - From the `airflow-scheduler` container, test connectivity to the ClusterIP:
    ```bash
    docker compose exec airflow-scheduler ping 10.96.0.100
    ```
    - If this fails, check network configuration and ensure the container is correctly connected to the `minikube` network.

- **Check Service Logs**:
  - Inspect the API service logs for connection attempts:
    ```bash
    kubectl logs -f deployment/health-predict-api-deployment
    ```
    - Look for errors or indications that the service is not accepting connections, which could suggest readiness issues.

- **Test Direct Socket Connection with ClusterIP**:
  - Modify `run_api_tests_callable` to include a direct socket test using the ClusterIP:
    ```python
    import socket
    cluster_ip = "10.96.0.100"  # Replace with actual ClusterIP
    port = 80
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((cluster_ip, port))
    if result == 0:
        logging.info("Direct socket connection to ClusterIP succeeded.")
    else:
        logging.error(f"Direct socket connection failed with error {result}")
    sock.close()
    ```
    - If this fails, it indicates a broader network issue within the container.

- **Check for Network Policies or Firewall Rules**:
  - Ensure no network policies or firewall rules in Minikube are blocking intra-cluster connections. Run:
    ```bash
    kubectl get networkpolicy
    ```
    - If policies exist, review them for restrictions affecting the Airflow container.

- **Compare Environment Variables**:
  - Print environment variables within `run_api_tests_callable` using `import os; print(os.environ)` and compare with `docker compose exec airflow-scheduler env`. Look for differences that might affect network access, though this is unlikely given shared network namespaces.

#### Summary of Findings

The analysis suggests that the networking issue is likely due to limitations in accessing NodePort services from Airflow's LocalExecutor subprocesses, possibly exacerbated by Minikube's Docker driver configuration on Linux. Modifying tests to use ClusterIP should resolve this by leveraging internal cluster networking, which is more reliable for intra-cluster communication. If issues persist, further investigation into network policies, firewall rules, and container connectivity is recommended.

This approach is supported by documentation on Minikube and Kubernetes service access, ensuring a robust solution for your MLOps pipeline's deployment testing phase.

---

### Key Citations
- [Minikube Accessing Apps Documentation](https://minikube.sigs.k8s.io/docs/handbook/accessing/)
- [Kubernetes Service Documentation](https://kubernetes.io/docs/concepts/services-networking/service/)