## 2025-05-14: Debugging CI/CD Docker Authentication and kubectl, Uncovered MLflow Model Issue

*   **Goal**: Resolve persistent "no basic auth credentials" for `docker push` in Airflow deployment DAG and ensure `kubectl` is available.
*   **Initial State**: DAG failing at `build_and_push_docker_image` due to Docker auth, and later at `update_kubernetes_deployment` due to `kubectl` not found.
*   **Actions & Observations**:
    *   Attempted "Fix A" (explicit `--config` for `docker push`): Failed, same auth error.
    *   Attempted "Fix B" (ensuring `HOME` and `DOCKER_CONFIG` env vars in `docker-compose.yml`): Failed, same auth error.
    *   Implemented "Fix C" (direct `aws ecr get-login-password ... | docker login ...` in a BashOperator):
        *   This successfully resolved the Docker authentication issue. The `ecr_login` and `build_and_push_docker_image` tasks passed.
    *   Addressed `kubectl: not found` error by adding `kubectl` installation to `mlops-services/Dockerfile.airflow`.
    *   Encountered Docker Compose build issues (`KeyError: 'ContainerConfig'`), resolved by a full Docker prune (`docker-compose down -v --remove-orphans && docker system prune -af && docker-compose up -d --build`).
    *   Discovered the `health_predict_api_deployment` DAG was paused, preventing runs from executing. Unpaused the DAG.
*   **Final State**: 
    *   DAG run `manual__2025-05-14T02:31:37+00:00` is now successfully executing past the Docker push and Kubernetes connection steps.
    *   The DAG now fails at the `get_production_model_info` task.
    *   **Error**: `mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Registered Model with name=HealthPredict_RandomForest not found`.
*   **Next Steps**: Investigate MLflow Model Registry to ensure `HealthPredict_RandomForest` model is registered and promoted to the `Production` stage, or update the DAG to use the correct model name/stage. 