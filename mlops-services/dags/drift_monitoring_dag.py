from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import pandas as pd
import boto3
import os
import logging
from typing import List, Dict, Any

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'health_predict_drift_monitoring',
    default_args=default_args,
    description='Monitor data drift and trigger retraining when needed',
    schedule_interval=timedelta(hours=6),  # Run every 6 hours for simulation
    start_date=datetime(2025, 6, 10),
    catchup=False,
    tags=['health-predict', 'drift-monitoring', 'phase-5'],
)

# Environment variables for drift monitoring
env_vars = {
    'S3_BUCKET_NAME': os.getenv('S3_BUCKET_NAME', 'health-predict-mlops-f9ac6509'),
    'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
    'DRIFT_MONITORING_EXPERIMENT': os.getenv('DRIFT_MONITORING_EXPERIMENT', 'HealthPredict_Drift_Monitoring'),
    'DRIFT_REPORTS_S3_PREFIX': os.getenv('DRIFT_REPORTS_S3_PREFIX', 'drift_monitoring/reports'),
    'DRIFT_BATCH_DATA_S3_PREFIX': os.getenv('DRIFT_BATCH_DATA_S3_PREFIX', 'drift_monitoring/batch_data'),
    'DRIFT_REFERENCE_DATA_S3_PREFIX': os.getenv('DRIFT_REFERENCE_DATA_S3_PREFIX', 'drift_monitoring/reference_data'),
    'DRIFT_THRESHOLD_MINOR': float(os.getenv('DRIFT_THRESHOLD_MINOR', '0.05')),
    'DRIFT_THRESHOLD_MODERATE': float(os.getenv('DRIFT_THRESHOLD_MODERATE', '0.15')),
    'DRIFT_THRESHOLD_MAJOR': float(os.getenv('DRIFT_THRESHOLD_MAJOR', '0.30')),
    'TARGET_COLUMN': 'readmitted_binary',  # Healthcare target column
    'BATCH_SIZE': 1000,  # Default batch size for simulation
}

def simulate_new_data_batch(**kwargs):
    """
    Simulate arrival of new data batch by creating a subset from future_data.csv
    This simulates the real-world scenario of new data arriving over time
    """
    logging.info("Starting data batch simulation...")
    
    s3_client = boto3.client('s3')
    bucket_name = env_vars['S3_BUCKET_NAME']
    
    # Generate batch identifier based on execution date
    execution_date = kwargs['execution_date']
    batch_id = execution_date.strftime("%Y%m%d_%H%M%S")
    batch_filename = f"batch_{batch_id}.csv"
    
    try:
        # Download future_data.csv to process
        future_data_key = 'processed_data/future_data.csv'
        local_future_data = f'/tmp/future_data_{batch_id}.csv'
        
        logging.info(f"Downloading {future_data_key} from S3...")
        s3_client.download_file(bucket_name, future_data_key, local_future_data)
        
        # Read the data and create a batch
        df = pd.read_csv(local_future_data)
        logging.info(f"Loaded future data with {len(df)} rows")
        
        # Calculate which batch to extract (based on execution count)
        # For simplicity, we'll use a hash of the batch_id to get a pseudo-random start position
        start_row = (hash(batch_id) % (len(df) - env_vars['BATCH_SIZE'])) if len(df) > env_vars['BATCH_SIZE'] else 0
        end_row = start_row + env_vars['BATCH_SIZE']
        
        # Extract the batch
        batch_data = df.iloc[start_row:end_row].copy()
        logging.info(f"Created batch with {len(batch_data)} rows (rows {start_row}-{end_row})")
        
        # Save batch locally and upload to S3
        local_batch_path = f'/tmp/{batch_filename}'
        batch_data.to_csv(local_batch_path, index=False)
        
        # Upload to S3 batch_data directory
        s3_batch_key = f"{env_vars['DRIFT_BATCH_DATA_S3_PREFIX']}/{batch_filename}"
        s3_client.upload_file(local_batch_path, bucket_name, s3_batch_key)
        
        logging.info(f"Uploaded batch to s3://{bucket_name}/{s3_batch_key}")
        
        # Clean up local files
        os.remove(local_future_data)
        os.remove(local_batch_path)
        
        # Return batch information for downstream tasks
        batch_info = {
            'batch_id': batch_id,
            'batch_filename': batch_filename,
            's3_batch_path': f"s3://{bucket_name}/{s3_batch_key}",
            'batch_size': len(batch_data),
            'start_row': start_row,
            'end_row': end_row
        }
        
        logging.info(f"Batch simulation completed: {batch_info}")
        return batch_info
        
    except Exception as e:
        logging.error(f"Error in data batch simulation: {str(e)}")
        raise

def detect_drift(**kwargs):
    """
    Run drift detection on the new batch using the monitor_drift.py script
    Returns drift status for downstream decision making
    """
    logging.info("Starting drift detection...")
    
    # Get batch info from upstream task
    batch_info = kwargs['ti'].xcom_pull(task_ids='simulate_data_batch')
    if not batch_info:
        raise ValueError("No batch information received from upstream task")
    
    logging.info(f"Running drift detection on batch: {batch_info['batch_id']}")
    
    # Construct S3 paths
    bucket_name = env_vars['S3_BUCKET_NAME']
    new_data_path = batch_info['s3_batch_path']
    reference_data_path = f"s3://{bucket_name}/{env_vars['DRIFT_REFERENCE_DATA_S3_PREFIX']}/initial_train.csv"
    reports_path = f"s3://{bucket_name}/{env_vars['DRIFT_REPORTS_S3_PREFIX']}/{batch_info['batch_id']}"
    
    # Construct the drift monitoring command
    drift_command = f"""
    python /opt/airflow/scripts/monitor_drift.py \
        --s3_new_data_path "{new_data_path}" \
        --s3_reference_data_path "{reference_data_path}" \
        --s3_evidently_reports_path "{reports_path}" \
        --mlflow_tracking_uri "{env_vars['MLFLOW_TRACKING_URI']}" \
        --mlflow_experiment_name "{env_vars['DRIFT_MONITORING_EXPERIMENT']}" \
        --target_column "{env_vars['TARGET_COLUMN']}" \
        --drift_threshold "{env_vars['DRIFT_THRESHOLD_MODERATE']}"
    """
    
    try:
        # Execute drift detection script
        import subprocess
        result = subprocess.run(
            drift_command.strip().split(),
            capture_output=True,
            text=True,
            check=False
        )
        
        # Parse the output to get drift status
        drift_status = result.stdout.strip().split('\n')[-1] if result.stdout else "DRIFT_MONITORING_ERROR"
        
        logging.info(f"Drift detection completed. Status: {drift_status}")
        logging.info(f"Command output: {result.stdout}")
        
        if result.stderr:
            logging.warning(f"Command stderr: {result.stderr}")
        
        # Return comprehensive drift information
        drift_info = {
            'batch_id': batch_info['batch_id'],
            'drift_status': drift_status,
            'drift_detected': drift_status == 'DRIFT_DETECTED',
            'reports_path': reports_path,
            'command_output': result.stdout,
            'return_code': result.returncode
        }
        
        return drift_info
        
    except Exception as e:
        logging.error(f"Error in drift detection: {str(e)}")
        drift_info = {
            'batch_id': batch_info['batch_id'],
            'drift_status': 'DRIFT_MONITORING_ERROR',
            'drift_detected': False,
            'error': str(e)
        }
        return drift_info

def evaluate_drift_severity(**kwargs):
    """
    Evaluate drift severity and determine appropriate response
    """
    logging.info("Evaluating drift severity...")
    
    drift_info = kwargs['ti'].xcom_pull(task_ids='run_drift_detection')
    if not drift_info:
        raise ValueError("No drift information received from upstream task")
    
    drift_status = drift_info['drift_status']
    batch_id = drift_info['batch_id']
    
    # TODO: In future steps, parse drift_share from command output for granular severity
    # For now, using simple binary drift detection
    
    if drift_status == 'DRIFT_DETECTED':
        # Assume moderate drift for now - will be enhanced in later steps
        severity = 'moderate'
        action = 'trigger_retraining'
        logging.info(f"Drift detected in batch {batch_id}. Severity: {severity}, Action: {action}")
    elif drift_status == 'NO_DRIFT':
        severity = 'none'
        action = 'continue_monitoring'
        logging.info(f"No drift detected in batch {batch_id}. Continue monitoring.")
    else:
        severity = 'unknown'
        action = 'investigate_error'
        logging.warning(f"Drift monitoring error for batch {batch_id}. Status: {drift_status}")
    
    severity_info = {
        'batch_id': batch_id,
        'drift_status': drift_status,
        'severity': severity,
        'recommended_action': action,
        'requires_retraining': action == 'trigger_retraining'
    }
    
    logging.info(f"Drift severity evaluation: {severity_info}")
    return severity_info

def decide_drift_response(**kwargs):
    """
    Branching function to decide which path to take based on drift evaluation
    """
    drift_severity = kwargs['ti'].xcom_pull(task_ids='evaluate_drift_severity')
    
    if not drift_severity:
        logging.error("No drift severity information available")
        return 'no_drift_continue_monitoring'
    
    action = drift_severity.get('recommended_action', 'continue_monitoring')
    severity = drift_severity.get('severity', 'none')
    
    logging.info(f"Drift decision: severity={severity}, action={action}")
    
    if action == 'trigger_retraining':
        return 'moderate_drift_prepare_retraining'
    elif action == 'continue_monitoring':
        return 'no_drift_continue_monitoring'
    else:
        # For errors or unknown states
        return 'no_drift_continue_monitoring'

# Task 1: Simulate new data batch arrival
simulate_data_batch = PythonOperator(
    task_id='simulate_data_batch',
    python_callable=simulate_new_data_batch,
    dag=dag,
)

# Task 2: Run drift detection on the new batch
run_drift_detection = PythonOperator(
    task_id='run_drift_detection',
    python_callable=detect_drift,
    dag=dag,
)

# Task 3: Evaluate drift severity and determine response
evaluate_drift = PythonOperator(
    task_id='evaluate_drift_severity',
    python_callable=evaluate_drift_severity,
    dag=dag,
)

# Task 4: Branching decision based on drift severity
drift_branch = BranchPythonOperator(
    task_id='decide_drift_response',
    python_callable=decide_drift_response,
    dag=dag,
)

# Task 5: Different drift response actions
no_drift_action = DummyOperator(
    task_id='no_drift_continue_monitoring',
    dag=dag,
)

moderate_drift_action = DummyOperator(
    task_id='moderate_drift_prepare_retraining',
    dag=dag,
)

# Task 6: Trigger retraining DAG for moderate/major drift
trigger_retraining = TriggerDagRunOperator(
    task_id='trigger_model_retraining',
    trigger_dag_id='health_predict_training_hpo',
    conf={
        'triggered_by': 'drift_monitoring',
        'drift_batch_id': '{{ ti.xcom_pull(task_ids="evaluate_drift_severity", key="return_value")["batch_id"] }}',
        'drift_severity': '{{ ti.xcom_pull(task_ids="evaluate_drift_severity", key="return_value")["severity"] }}',
    },
    dag=dag,
)

# Task 7: Log drift monitoring completion
log_completion = PythonOperator(
    task_id='log_monitoring_completion',
    python_callable=lambda **kwargs: logging.info(
        f"Drift monitoring cycle completed for batch: "
        f"{kwargs['ti'].xcom_pull(task_ids='evaluate_drift_severity')['batch_id']}"
    ),
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# Define task dependencies with proper branching
simulate_data_batch >> run_drift_detection >> evaluate_drift >> drift_branch

# Branching paths
drift_branch >> no_drift_action >> log_completion
drift_branch >> moderate_drift_action >> trigger_retraining >> log_completion 