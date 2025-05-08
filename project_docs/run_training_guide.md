# Guide: Running the Model Training Script (`scripts/train_model.py`)

This guide provides step-by-step instructions to execute the model training script. The script runs inside a Docker container managed by Docker Compose, ensuring a reproducible environment. It performs hyperparameter optimization (HPO) using Ray Tune and logs experiments, models, and artifacts to MLflow, with artifacts being stored in S3.

## Prerequisites:

1.  **EC2 Instance Running & Docker Services:** Your AWS EC2 instance should be running. Docker and Docker Compose should be installed, and your MLOps services (`postgres`, `mlflow`, `jupyterlab`, etc.) should be defined in `~/health-predict/mlops-services/docker-compose.yml` and ideally already running (`docker-compose -f ~/health-predict/mlops-services/docker-compose.yml up -d --build`).
2.  **Project Cloned:** The `health-predict` project repository should be cloned into the `~/health-predict` directory on your EC2 instance.
3.  **S3 Data:** The initial training and validation datasets (`initial_train.csv`, `initial_validation.csv`) must be present in your S3 bucket (`s3://health-predict-mlops-f9ac6509/processed_data/`).
4.  **Dependencies Defined:** The Python dependencies for the training script must be listed in `~/health-predict/scripts/requirements-training.txt`. The `jupyterlab` service in your `docker-compose.yml` should be configured to install these dependencies.

## Steps to Run the Training Script:

1.  **SSH into your EC2 Instance:**
    Connect to your EC2 instance using SSH.

2.  **Navigate to the MLOps Services Directory:**
    Open your terminal and change to the directory containing your `docker-compose.yml` file:
    ```bash
    cd ~/health-predict/mlops-services
    ```

3.  **Ensure Docker Services are Up-to-Date and Running:**
    If you've recently updated `docker-compose.yml` (e.g., to include the `pip install` step for `jupyterlab`) or `scripts/requirements-training.txt`, you **must rebuild and restart** the relevant service(s). To be safe, rebuild and restart all services:
    ```bash
    docker-compose up -d --build
    ```
    Wait for the services, especially `jupyterlab`, to build and start. The first time after adding the `pip install` command, the `jupyterlab` service might take a few minutes longer to start as it downloads and installs the packages.
    You can check the status with `docker-compose ps` and logs with `docker-compose logs jupyterlab`.

4.  **Construct and Execute the Training Command (inside Docker):**
    You will need your EC2 instance's current **Public IP Address** to form the MLflow tracking URI.

    The training script will be executed *inside* the `jupyterlab` container. The project directory (`~/health-predict`) is mounted as `/home/jovyan/work` inside this container.

    Execute the following command from the `~/health-predict/mlops-services` directory (where your `docker-compose.yml` is):

    ```bash
    docker-compose exec jupyterlab python /home/jovyan/work/scripts/train_model.py \
        --s3-bucket-name health-predict-mlops-f9ac6509 \
        --train-key processed_data/initial_train.csv \
        --validation-key processed_data/initial_validation.csv \
        --mlflow-tracking-uri http://<YOUR_EC2_PUBLIC_IP>:5000 \
        --mlflow-experiment-name "HealthPredict_Training_HPO" \
        --ray-num-samples 10 \
        --ray-max-epochs-per-trial 10 \
        --ray-grace-period 1 \
        --ray-local-dir "/home/jovyan/work/ray_results_training_run" # Path inside the container
    ```

    **Explanation of Command Arguments:**
    *   `docker-compose exec jupyterlab`: This tells Docker Compose to execute a command inside the running `jupyterlab` service container.
    *   `python /home/jovyan/work/scripts/train_model.py`: The command to run inside the container. Note the path to the script reflecting the mount point.
    *   `--s3-bucket-name`, `--train-key`, `--validation-key`: Same as before.
    *   `--mlflow-tracking-uri`: The address of your MLflow tracking server. **Replace `<YOUR_EC2_PUBLIC_IP>` with the current public IP of your EC2 instance.** The MLflow service is accessible from within the Docker network by its service name (`mlflow`), so an alternative for scripts run *within* Docker could be `http://mlflow:5000`. However, for Ray Tune workers that might not resolve service names the same way (depending on Ray's networking setup with Docker), using the EC2 public IP is often more reliable for externally accessible services like MLflow UI, even if components are on the same host.
    *   `--mlflow-experiment-name`: Same as before.
    *   `--ray-num-samples`, `--ray-max-epochs-per-trial`, `--ray-grace-period`: Same as before.
    *   `--ray-local-dir "/home/jovyan/work/ray_results_training_run"`: Local directory *inside the container* where Ray Tune will store its results. Since `/home/jovyan/work` is mounted from your project root, these results will appear in `~/health-predict/ray_results_training_run` on your EC2 host.

5.  **Monitor Script Execution:**
    The script will output logs to your terminal where you ran the `docker-compose exec` command. You should observe messages related to data loading, preprocessing, Ray Tune HPO trials, and MLflow logging.
    This process can take a significant amount of time.

6.  **Verify Results in MLflow UI:**
    *   Once the script execution is complete, open a web browser and navigate to your MLflow UI using your EC2 instance's public IP: `http://<YOUR_EC2_PUBLIC_IP>:5000`.
    *   Follow the verification steps outlined in the previous version of this guide to check for the `Preprocessing_Run`, HPO trial runs, and `Best_<ModelType>_Model` runs, along with their artifacts in S3.

## Discovering the Best Performing Model:

*   As before, use the MLflow UI to compare validation metrics of the `Best_<ModelType>_Model` runs to identify the overall best model.

If you encounter any issues, review the terminal output from the `docker-compose exec` command. You can also check the logs of the `jupyterlab` service itself using `docker-compose logs jupyterlab` (from the `~/health-predict/mlops-services` directory). 