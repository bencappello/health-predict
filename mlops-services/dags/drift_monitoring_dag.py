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
            's3_batch_path': s3_batch_key,  # Store the S3 key for drift context
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
    new_data_path = f"s3://{bucket_name}/{batch_info['s3_batch_path']}"
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
            'batch_path': batch_info['s3_batch_path'],
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
    Evaluate drift severity using graduated response system
    """
    logging.info("Evaluating drift severity with graduated response system...")
    
    drift_info = kwargs['ti'].xcom_pull(task_ids='run_drift_detection')
    if not drift_info:
        raise ValueError("No drift information received from upstream task")
    
    batch_id = drift_info['batch_id']
    
    # Import graduated response handler
    import sys
    sys.path.append('/opt/airflow/scripts')
    from drift_response_handler import DriftResponseHandler, DriftMetrics
    
    try:
        # Parse drift metrics from command output
        command_output = drift_info.get('command_output', '')
        drift_metrics = parse_drift_metrics_from_output(command_output)
        
        # Create drift metrics object
        metrics = DriftMetrics(
            dataset_drift_score=drift_metrics.get('dataset_drift_score', 0.0),
            feature_drift_count=drift_metrics.get('feature_drift_count', 0),
            total_features=drift_metrics.get('total_features', 1),
            concept_drift_score=drift_metrics.get('concept_drift_score'),
            prediction_drift_score=drift_metrics.get('prediction_drift_score'),
            performance_degradation=drift_metrics.get('performance_degradation'),
            confidence_score=drift_metrics.get('drift_confidence_score', 0.5)
        )
        
        # Get drift context
        drift_context = {
            'batch_id': batch_id,
            'execution_date': kwargs['execution_date'].isoformat(),
            'last_retraining_timestamp': get_last_retraining_timestamp(),
            'consecutive_major_drift_count': get_consecutive_drift_count(),
            'batch_path': drift_info.get('batch_path', ''),
            'reports_path': drift_info.get('reports_path', '')
        }
        
        # Initialize handler and evaluate response
        handler = DriftResponseHandler()
        response = handler.evaluate_drift_response(metrics, drift_context)
        
        # Convert response to serializable format
        severity_info = {
            'batch_id': batch_id,
            'drift_status': drift_info['drift_status'],
            'severity': response.severity.value,
            'action': response.action.value,
            'confidence': response.confidence,
            'reasoning': response.reasoning,
            'recommendations': response.recommendations,
            'escalation_needed': response.escalation_needed,
            'requires_retraining': response.action.value in ['incremental_retrain', 'full_retrain'],
            'drift_context': drift_context,
            'drift_metrics': drift_metrics
        }
        
        logging.info(f"Graduated drift response: {response.severity.value} -> {response.action.value}")
        logging.info(f"Confidence: {response.confidence:.2f}, Reasoning: {response.reasoning}")
        
        return severity_info
        
    except Exception as e:
        logging.error(f"Error in graduated drift evaluation: {e}")
        # Fallback to simple evaluation
        drift_status = drift_info['drift_status']
        
        if drift_status == 'DRIFT_DETECTED':
            severity = 'moderate'
            action = 'incremental_retrain'
        elif drift_status == 'NO_DRIFT':
            severity = 'none'
            action = 'continue_monitoring'
        else:
            severity = 'unknown'
            action = 'continue_monitoring'
        
        return {
            'batch_id': batch_id,
            'drift_status': drift_status,
            'severity': severity,
            'action': action,
            'confidence': 0.5,
            'reasoning': f"Fallback evaluation due to error: {e}",
            'requires_retraining': action in ['incremental_retrain', 'full_retrain'],
            'error': str(e)
        }

def parse_drift_metrics_from_output(command_output: str) -> dict:
    """
    Parse drift metrics from monitor_drift.py command output
    """
    metrics = {
        'dataset_drift_score': 0.0,
        'feature_drift_count': 0,
        'total_features': 1,
        'drift_confidence_score': 0.5
    }
    
    try:
        lines = command_output.split('\n')
        for line in lines:
            if 'Dataset drift score:' in line:
                metrics['dataset_drift_score'] = float(line.split(':')[-1].strip())
            elif 'Features with drift:' in line:
                metrics['feature_drift_count'] = int(line.split(':')[-1].strip().split()[0])
            elif 'Total features:' in line:
                metrics['total_features'] = int(line.split(':')[-1].strip())
            elif 'Drift confidence:' in line:
                metrics['drift_confidence_score'] = float(line.split(':')[-1].strip())
    except Exception as e:
        logging.warning(f"Error parsing drift metrics: {e}")
    
    return metrics

def get_last_retraining_timestamp() -> float:
    """
    Get timestamp of last retraining event from MLflow
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(env_vars['MLFLOW_TRACKING_URI'])
        
        # Search for recent training runs
        experiment = mlflow.get_experiment_by_name('HealthPredict_Training_HPO_Airflow')
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs.empty:
                return runs.iloc[0]['start_time'].timestamp()
    except Exception as e:
        logging.warning(f"Error getting last retraining timestamp: {e}")
    
    return 0.0

def get_consecutive_drift_count() -> int:
    """
    Get count of consecutive drift events (simplified implementation)
    """
    # TODO: Implement proper drift history tracking
    # For now, return 0 to avoid unnecessary escalations
    return 0

def decide_drift_response(**kwargs):
    """
    Enhanced branching function with graduated response logic
    """
    drift_evaluation = kwargs['ti'].xcom_pull(task_ids='evaluate_drift_severity')
    
    if not drift_evaluation:
        logging.error("No drift evaluation information available")
        return 'no_drift_continue_monitoring'
    
    action = drift_evaluation.get('action', 'continue_monitoring')
    severity = drift_evaluation.get('severity', 'none')
    
    logging.info(f"Graduated drift decision: severity={severity}, action={action}")
    
    # Map graduated response actions to DAG tasks
    if action == 'continue_monitoring':
        return 'no_drift_continue_monitoring'
    elif action == 'log_and_monitor':
        return 'minor_drift_log_and_monitor' 
    elif action in ['incremental_retrain', 'full_retrain']:
        return 'drift_trigger_retraining'
    elif action == 'architecture_review':
        return 'concept_drift_alert'
    else:
        # For unknown states, default to monitoring
        logging.warning(f"Unknown action '{action}', defaulting to monitoring")
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

# Task 5: Graduated drift response actions
no_drift_action = PythonOperator(
    task_id='no_drift_continue_monitoring',
    python_callable=lambda **kwargs: logging.info("No drift detected - continuing normal monitoring"),
    dag=dag,
)

minor_drift_action = PythonOperator(
    task_id='minor_drift_log_and_monitor',
    python_callable=lambda **kwargs: logging.warning(
        f"Minor drift detected in batch {kwargs['ti'].xcom_pull(task_ids='evaluate_drift_severity')['batch_id']} - "
        f"increased monitoring activated"
    ),
    dag=dag,
)

concept_drift_action = PythonOperator(
    task_id='concept_drift_alert',
    python_callable=lambda **kwargs: logging.critical(
        f"Concept drift detected in batch {kwargs['ti'].xcom_pull(task_ids='evaluate_drift_severity')['batch_id']} - "
        f"manual architecture review required"
    ),
    dag=dag,
)

# Task 6: Enhanced retraining trigger with full drift context
trigger_retraining = TriggerDagRunOperator(
    task_id='drift_trigger_retraining',
    trigger_dag_id='health_predict_training_hpo',
    conf={
        'drift_triggered': True,
        'triggered_by': 'drift_monitoring',
        'drift_batch_id': '{{ ti.xcom_pull(task_ids="evaluate_drift_severity")["batch_id"] }}',
        'drift_severity': '{{ ti.xcom_pull(task_ids="evaluate_drift_severity")["severity"] }}',
        'drift_action': '{{ ti.xcom_pull(task_ids="evaluate_drift_severity")["action"] }}',
        'drift_confidence': '{{ ti.xcom_pull(task_ids="evaluate_drift_severity")["confidence"] }}',
        'drift_reasoning': '{{ ti.xcom_pull(task_ids="evaluate_drift_severity")["reasoning"] }}',
        'drift_batch_path': '{{ ti.xcom_pull(task_ids="simulate_data_batch")["s3_batch_path"] }}',
        'drift_reports_path': '{{ ti.xcom_pull(task_ids="run_drift_detection")["reports_path"] }}',
        'retraining_timestamp': '{{ ts }}'
    },
    wait_for_completion=False,  # Don't wait for training to complete
    poke_interval=30,
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

# Define task dependencies with graduated response branching
simulate_data_batch >> run_drift_detection >> evaluate_drift >> drift_branch

# Graduated response branching paths
drift_branch >> no_drift_action >> log_completion
drift_branch >> minor_drift_action >> log_completion  
drift_branch >> concept_drift_action >> log_completion
drift_branch >> trigger_retraining >> log_completion 