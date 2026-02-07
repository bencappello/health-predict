# API Test Connectivity Debugging Strategy

This document outlines a strategy to debug and resolve the "Connection Refused" errors encountered by the `run_api_tests` task in the `health_predict_api_deployment` Airflow DAG. The primary goal is to enable `pytest` (executed via a PythonOperator) to successfully connect to the FastAPI service running in Minikube.

## Background

Current situation:
- The FastAPI service is deployed to Minikube via a Kubernetes Deployment and exposed via a NodePort service.
- The Airflow services (scheduler, webserver) run in Docker containers managed by Docker Compose.
- These Airflow containers are connected to the `minikube` Docker network.
- `curl` commands from within the `airflow-scheduler` container to the service (via Minikube Node IP and NodePort, e.g., `192.168.49.2:<NodePort>`) succeed.
- Python scripts (`pytest` or direct `socket.connect_ex`) executed by the Airflow PythonOperator fail with "Connection Refused" when trying to reach the same NodePort.
- The `run_api_tests` task in `deployment_pipeline_dag.py` currently skips the actual tests due to this issue.

## Strategy

The strategy will proceed in phases, starting with the most K8s-native and least invasive approach, then moving to broader networking diagnostics and solutions if needed.

### Phase 1: Implement ClusterIP-Based Testing (Primary Approach)

This approach leverages Kubernetes internal service discovery, assuming the Airflow container's connection to the `minikube` network is sufficient for ClusterIP access.

**Rationale**: Accessing services via their ClusterIP and service port is the standard way for components within the same Kubernetes (or Minikube emulated) network to communicate. This bypasses potential complexities or restrictions related to NodePort access from within certain container execution contexts.

**Steps**:

1.  **Verify Airflow Container Network Configuration**:
    *   **Action**: Confirm that the Airflow services in `mlops-services/docker-compose.yml` are indeed connected to the `minikube` network.
        ```yaml
        # Example snippet from docker-compose.yml
        services:
          airflow-scheduler:
            networks:
              - default # or your primary app network
              - minikube # Ensure this is present and correctly defined
        # ...
        networks:
          minikube:
            external: true # Or however it's configured to connect to the Minikube Docker network
        ```
    *   **Check**: Inspect a running Airflow container (`docker inspect <airflow_scheduler_container_id>`) to see its network memberships and ensure it's part of the network that Minikube uses.

2.  **Dynamically Obtain Service ClusterIP and Port**:
    *   **Action**: Modify `mlops-services/dags/deployment_pipeline_dag.py` in the `construct_test_command` function.
    *   **Details**:
        *   Use `subprocess.check_output` to execute `kubectl get svc health-predict-api-service -o jsonpath='{.spec.clusterIP}'` to get the ClusterIP.
        *   Use `subprocess.check_output` to execute `kubectl get svc health-predict-api-service -o jsonpath='{.spec.ports[0].port}'` to get the service port (the port the service listens on, not the `targetPort` or `nodePort` directly, though often same as `targetPort` if not specified otherwise). The API pod listens on port 8000.
    *   **Error Handling**: Add robust error handling for these `kubectl` calls.

3.  **Update Test Script (`tests/api/test_api_endpoints.py`)**:
    *   **Action**: Modify the script to construct `API_BASE_URL` using environment variables that will be set by the DAG.
    *   **Details**:
        *   Expect new environment variables like `API_CLUSTER_IP` and `API_SERVICE_PORT`.
        *   `API_BASE_URL = f"http://{os.getenv('API_CLUSTER_IP')}:{os.getenv('API_SERVICE_PORT')}"`
        *   Ensure `api_session.trust_env = False` remains in place if proxies were a concern, though ClusterIPs should typically be exempt from proxying.

4.  **Update DAG Task `construct_test_command`**:
    *   **Action**: Modify `mlops-services/dags/deployment_pipeline_dag.py`.
    *   **Details**:
        *   Pass the fetched ClusterIP and service port as environment variables (e.g., `API_CLUSTER_IP`, `API_SERVICE_PORT`) in the `env_block` for the `pytest` command.
        *   Example: `env_block = f"API_CLUSTER_IP='{cluster_ip}' API_SERVICE_PORT='{service_port}'"`

5.  **Enable Tests in `run_api_tests_callable`**:
    *   **Action**: Modify `mlops-services/dags/deployment_pipeline_dag.py`.
    *   **Details**: Remove the current workaround that skips test execution. The function should now:
        *   Pull the `test_command` from XCom.
        *   Execute it using `subprocess.run`.
        *   Check the return code and raise `AirflowFailException` if tests fail.

6.  **Test and Iterate**:
    *   **Action**: Trigger the `health_predict_api_deployment` DAG.
    *   **Monitoring**:
        *   Closely monitor the logs of the `construct_api_test_command` and `run_api_tests` tasks.
        *   Check for successful `kubectl` calls and correct environment variable propagation.
        *   If tests fail, examine `pytest` output for specific connection errors.

### Phase 2: Enhanced Diagnostics (If ClusterIP Fails)

If Phase 1 does not resolve the issue, perform these diagnostic steps from within the Airflow worker's execution context.

1.  **Basic Connectivity Tests in `run_api_tests_callable`**:
    *   **Action**: Before running `pytest`, add diagnostic Python code to `run_api_tests_callable`:
        *   Log the fetched `API_CLUSTER_IP` and `API_SERVICE_PORT`.
        *   Attempt `ping <API_CLUSTER_IP>` using `subprocess.run(["ping", "-c", "3", api_cluster_ip])`. Log output.
        *   Attempt `curl http://<API_CLUSTER_IP>:<API_SERVICE_PORT>/health` using `subprocess.run(["curl", "-v", url])`. Log output.
        *   Attempt a direct Python socket connection:
            ```python
            import socket
            import errno
            # ...
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5) # 5 second timeout
            result = sock.connect_ex((api_cluster_ip, int(api_service_port)))
            if result == 0:
                logging.info(f"Direct socket connection to {api_cluster_ip}:{api_service_port} SUCCEEDED.")
            else:
                logging.error(f"Direct socket connection to {api_cluster_ip}:{api_service_port} FAILED with errno {result}: {os.strerror(result)}")
            sock.close()
            ```
    *   **Analysis**: Compare these results with manual `docker compose exec airflow-scheduler ...` commands.

2.  **Examine API Pod Logs**:
    *   **Action**: `kubectl logs -l app=health-predict-api-deployment -c health-predict-api-container --tail=100 -f` during test execution.
    *   **Check**: Look for any incoming connection attempts, successes, or errors from the ClusterIP.

3.  **Check Kubernetes Network Policies**:
    *   **Action**: `kubectl get networkpolicy --all-namespaces`
    *   **Check**: Ensure no policies are inadvertently blocking traffic from the Airflow pod's source IP range (if identifiable) to the API service.

4.  **Review Proxy Settings Thoroughly**:
    *   **Action**: Within `run_api_tests_callable`, print all proxy-related environment variables (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`).
    *   **Check**: Ensure `NO_PROXY` correctly includes the ClusterIP range (e.g., `10.0.0.0/8` or the specific Minikube service CIDR) if any proxies are set. The `api_session.trust_env = False` in `test_api_endpoints.py` should handle this for `requests`, but it's good to verify the environment.

### Phase 3: Broader Networking Solutions (If Diagnostics Point to Deeper Issues)

These are more involved solutions, drawing from `gpt_dag_test_suggestion.md`.

1.  **Minikube Service Tunnel / `kubectl port-forward`**:
    *   **Concept**: Forward the Minikube service to a port on the EC2 host's loopback interface (`127.0.0.1`) or `host.docker.internal`, then have the Airflow task connect to that.
    *   **`kubectl port-forward` (preferred for control from DAG)**:
        *   **Action**: In the DAG, before `run_api_tests_task`, add a `BashOperator` or Python function to start `kubectl port-forward svc/health-predict-api-service <chosen_local_port>:SERVICE_PORT` in the background. Ensure it's killed afterward.
        *   **Connection**: Tests connect to `http://host.docker.internal:<chosen_local_port>` or `http://127.0.0.1:<chosen_local_port>` (if `host.docker.internal` is problematic, use the EC2 host's IP that the Airflow container can reach).
        *   **Complexity**: Managing the lifecycle of the port-forward process.
    *   Requires `host.docker.internal` to be correctly resolved by the Airflow containers or an appropriate host IP to be used. Add to `extra_hosts` in `docker-compose.yml` for Airflow services if needed: `host.docker.internal:host-gateway`.

2.  **Review `docker network connect`**:
    *   **Action**: While `docker-compose.yml` should handle connecting Airflow to the `minikube` network if configured with `external: true`, double-check the runtime status.
    *   **Tactics**: Manually run `docker network inspect minikube` and `docker inspect <airflow_container_id>` to verify they share the network and subnet. This is more of a verification of the existing setup.

### Phase 4: Long-Term Improvements

1.  **Airflow Sensor for Service Readiness**:
    *   **Action**: Implement an `HttpSensor` or a custom sensor in the deployment DAG before the `run_api_tests` task to poll the API's `/health` endpoint until it's responsive.
    *   **Benefit**: Ensures tests only run when the API is confirmed to be live, avoiding timing-related connection issues.

2.  **Consider KubernetesExecutor for Airflow**:
    *   **Action**: For future iterations or more complex K8s interactions, evaluate migrating Airflow to run with the `KubernetesExecutor`.
    *   **Benefit**: Airflow tasks would run as K8s pods directly within the cluster, simplifying network access to other cluster services. This is a larger architectural change.

## Documentation and Journaling

- All changes, experiments, and outcomes will be meticulously logged in `project_docs/ai_docs/project_work_journal.md`.
- This strategy document will be updated as new information is found or if the plan needs to pivot.

This structured approach should help systematically identify and resolve the connectivity issue. 