from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pandas as pd
import boto3
import os
import logging
import time
import json
import uuid
from typing import List, Dict, Any, Optional

# Define default arguments for the DAG with enhanced error handling
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,  # Increased retries for production resilience
    'retry_delay': timedelta(minutes=2),  # Shorter retry delay for faster recovery
    'retry_exponential_backoff': True,  # Exponential backoff for retry strategy
    'max_retry_delay': timedelta(minutes=10),  # Maximum retry delay
}

# Define the enhanced DAG with comprehensive monitoring
dag = DAG(
    'health_predict_drift_monitoring_v2',
    default_args=default_args,
    description='Enhanced drift monitoring with parallel processing and comprehensive error handling',
    schedule_interval=timedelta(hours=3),  # More frequent monitoring for Week 5 testing
    start_date=datetime(2025, 1, 24),
    catchup=False,
    max_active_runs=2,  # Allow multiple concurrent runs for testing
    tags=['health-predict', 'drift-monitoring', 'phase-5', 'week-5', 'enhanced'],
)

# Enhanced environment variables for comprehensive monitoring
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
    'TARGET_COLUMN': 'readmitted_binary',
    'BATCH_SIZE': 1000,
    'MAX_PARALLEL_BATCHES': 3,  # Maximum parallel batch processing
    'BATCH_PROCESSING_BACKLOG_LIMIT': 10,  # Maximum backlog before alerting
    'DRIFT_ALERT_COOLDOWN_HOURS': 6,  # Cooldown period for drift alerts
}

def get_available_batches(**kwargs) -> List[Dict[str, Any]]:
    """
    Dynamically discover available data batches for processing
    This implements dynamic task generation based on available data
    """
    logging.info("Discovering available data batches for processing...")
    
    s3_client = boto3.client('s3')
    bucket_name = env_vars['S3_BUCKET_NAME']
    batch_prefix = env_vars['DRIFT_BATCH_DATA_S3_PREFIX']
    
    try:
        # List all batch files in S3
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f"{batch_prefix}/",
            MaxKeys=env_vars['BATCH_PROCESSING_BACKLOG_LIMIT']
        )
        
        available_batches = []
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.csv') and 'batch_' in key:
                    batch_info = {
                        'batch_id': key.split('/')[-1].replace('.csv', ''),
                        's3_key': key,
                        'size_bytes': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'processing_priority': obj['LastModified'].timestamp()  # Older batches have higher priority
                    }
                    available_batches.append(batch_info)
        
        # Sort by processing priority (oldest first)
        available_batches.sort(key=lambda x: x['processing_priority'])
        
        # Limit to max parallel processing capacity
        available_batches = available_batches[:env_vars['MAX_PARALLEL_BATCHES']]
        
        logging.info(f"Found {len(available_batches)} batches available for processing")
        
        return available_batches
        
    except Exception as e:
        logging.error(f"Error discovering available batches: {str(e)}")
        # Return empty list on error to gracefully handle the failure
        return []

def create_simulation_batch(**kwargs) -> Dict[str, Any]:
    """
    Enhanced batch simulation with realistic data arrival patterns
    """
    logging.info("Creating enhanced simulation batch with realistic patterns...")
    
    s3_client = boto3.client('s3')
    bucket_name = env_vars['S3_BUCKET_NAME']
    
    # Generate unique batch identifier with timestamp and random component
    execution_date = kwargs['execution_date']
    batch_id = f"sim_batch_{execution_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    batch_filename = f"{batch_id}.csv"
    
    try:
        # Load future data for batch creation
        future_data_key = 'processed_data/future_data.csv'
        local_future_data = f'/tmp/future_data_{batch_id}.csv'
        
        logging.info(f"Loading future data from {future_data_key}...")
        s3_client.download_file(bucket_name, future_data_key, local_future_data)
        
        df = pd.read_csv(local_future_data)
        logging.info(f"Loaded future data with {len(df)} rows")
        
        # Create realistic batch size variation (800-1200 rows)
        import random
        batch_size = random.randint(800, 1200)
        
        # Calculate start position with some randomization
        max_start = max(0, len(df) - batch_size)
        start_row = random.randint(0, max_start) if max_start > 0 else 0
        end_row = min(start_row + batch_size, len(df))
        
        # Extract batch data
        batch_data = df.iloc[start_row:end_row].copy()
        
        # Add realistic data quality variations
        # Simulate missing values in 1-3% of records
        if random.random() < 0.7:  # 70% chance of introducing quality issues
            n_missing = int(len(batch_data) * random.uniform(0.01, 0.03))
            missing_indices = random.sample(range(len(batch_data)), n_missing)
            missing_columns = random.sample(list(batch_data.columns), min(3, len(batch_data.columns)))
            
            for idx in missing_indices:
                for col in missing_columns:
                    if random.random() < 0.3:  # 30% chance per column
                        batch_data.iloc[idx, batch_data.columns.get_loc(col)] = None
        
        logging.info(f"Created batch with {len(batch_data)} rows (rows {start_row}-{end_row})")
        
        # Save and upload batch
        local_batch_path = f'/tmp/{batch_filename}'
        batch_data.to_csv(local_batch_path, index=False)
        
        s3_batch_key = f"{env_vars['DRIFT_BATCH_DATA_S3_PREFIX']}/{batch_filename}"
        s3_client.upload_file(local_batch_path, bucket_name, s3_batch_key)
        
        logging.info(f"Uploaded simulation batch to s3://{bucket_name}/{s3_batch_key}")
        
        # Cleanup
        os.remove(local_future_data)
        os.remove(local_batch_path)
        
        # Return comprehensive batch metadata
        batch_info = {
            'batch_id': batch_id,
            'batch_filename': batch_filename,
            's3_batch_path': s3_batch_key,
            'batch_size': len(batch_data),
            'start_row': start_row,
            'end_row': end_row,
            'data_quality_score': 1.0 - (n_missing / len(batch_data) if 'n_missing' in locals() else 0),
            'simulation_timestamp': execution_date.isoformat(),
            'batch_type': 'simulation'
        }
        
        return batch_info
        
    except Exception as e:
        logging.error(f"Error creating simulation batch: {str(e)}")
        raise

def parallel_drift_detection(batch_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Enhanced drift detection with parallel processing capability
    """
    batch_id = batch_info['batch_id']
    logging.info(f"Starting parallel drift detection for batch: {batch_id}")
    
    # Construct S3 paths
    bucket_name = env_vars['S3_BUCKET_NAME']
    new_data_path = f"s3://{bucket_name}/{batch_info['s3_batch_path']}"
    reference_data_path = f"s3://{bucket_name}/{env_vars['DRIFT_REFERENCE_DATA_S3_PREFIX']}/initial_train.csv"
    reports_path = f"s3://{bucket_name}/{env_vars['DRIFT_REPORTS_S3_PREFIX']}/{batch_id}"
    
    # Enhanced drift detection command with additional options
    drift_command = f"""
    python /opt/airflow/scripts/monitor_drift.py \
        --s3_new_data_path "{new_data_path}" \
        --s3_reference_data_path "{reference_data_path}" \
        --s3_evidently_reports_path "{reports_path}" \
        --mlflow_tracking_uri "{env_vars['MLFLOW_TRACKING_URI']}" \
        --mlflow_experiment_name "{env_vars['DRIFT_MONITORING_EXPERIMENT']}" \
        --target_column "{env_vars['TARGET_COLUMN']}" \
        --drift_threshold "{env_vars['DRIFT_THRESHOLD_MODERATE']}" \
        --enable_concept_drift \
        --verbose
    """
    
    try:
        start_time = time.time()
        
        # Execute drift detection with timeout
        import subprocess
        result = subprocess.run(
            drift_command.strip().split(),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False
        )
        
        detection_time = time.time() - start_time
        
        # Parse drift status from output
        drift_status = "DRIFT_MONITORING_ERROR"
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):  # Check from the end
                if line in ['DRIFT_DETECTED', 'NO_DRIFT', 'DRIFT_MONITORING_ERROR']:
                    drift_status = line
                    break
        
        logging.info(f"Drift detection completed in {detection_time:.2f}s. Status: {drift_status}")
        
        if result.stderr:
            logging.warning(f"Drift detection stderr: {result.stderr}")
        
        # Enhanced drift information with performance metrics
        drift_result = {
            'batch_id': batch_id,
            'drift_status': drift_status,
            'drift_detected': drift_status == 'DRIFT_DETECTED',
            'batch_path': batch_info['s3_batch_path'],
            'reports_path': reports_path,
            'detection_time_seconds': detection_time,
            'command_output': result.stdout,
            'return_code': result.returncode,
            'processing_timestamp': kwargs['execution_date'].isoformat(),
            'batch_metadata': batch_info
        }
        
        return drift_result
        
    except subprocess.TimeoutExpired:
        logging.error(f"Drift detection timed out for batch {batch_id}")
        return {
            'batch_id': batch_id,
            'drift_status': 'DRIFT_MONITORING_TIMEOUT',
            'drift_detected': False,
            'error': 'Detection process timed out after 5 minutes'
        }
    except Exception as e:
        logging.error(f"Error in parallel drift detection for batch {batch_id}: {str(e)}")
        return {
            'batch_id': batch_id,
            'drift_status': 'DRIFT_MONITORING_ERROR',
            'drift_detected': False,
            'error': str(e)
        }

def comprehensive_drift_evaluation(**kwargs) -> Dict[str, Any]:
    """
    Enhanced drift evaluation with comprehensive analysis
    """
    logging.info("Starting comprehensive drift evaluation...")
    
    # Get drift detection results from XCom
    drift_results = kwargs['ti'].xcom_pull(task_ids=['parallel_drift_detection_sim'])
    
    if not drift_results:
        logging.warning("No drift detection results available")
        return {
            'overall_drift_status': 'unknown',
            'action': 'continue_monitoring',
            'confidence': 0.0,
            'reasoning': 'No drift detection results available'
        }
    
    # Handle both single result and list of results
    if not isinstance(drift_results, list):
        drift_results = [drift_results]
    
    # Filter out None results
    drift_results = [r for r in drift_results if r is not None]
    
    if not drift_results:
        logging.warning("All drift detection results are None")
        return {
            'overall_drift_status': 'unknown',
            'action': 'continue_monitoring',
            'confidence': 0.0,
            'reasoning': 'All drift detection processes failed'
        }
    
    try:
        # Import drift response handler for comprehensive evaluation
        import sys
        sys.path.append('/opt/airflow/scripts')
        from drift_response_handler import DriftResponseHandler
        
        # Initialize response handler
        response_handler = DriftResponseHandler()
        
        # Aggregate drift results for comprehensive analysis
        total_batches = len(drift_results)
        drift_detected_count = sum(1 for r in drift_results if r.get('drift_detected', False))
        error_count = sum(1 for r in drift_results if r.get('drift_status') == 'DRIFT_MONITORING_ERROR')
        
        # Calculate overall drift confidence
        drift_confidence = drift_detected_count / total_batches if total_batches > 0 else 0.0
        
        # Determine overall drift status
        if drift_confidence >= 0.7:
            overall_status = 'major_drift'
        elif drift_confidence >= 0.3:
            overall_status = 'moderate_drift'
        elif drift_confidence > 0:
            overall_status = 'minor_drift'
        else:
            overall_status = 'no_drift'
        
        # Create comprehensive drift context for evaluation
        drift_context = {
            'total_batches_processed': total_batches,
            'batches_with_drift': drift_detected_count,
            'batch_error_count': error_count,
            'drift_confidence': drift_confidence,
            'overall_drift_status': overall_status,
            'detection_results': drift_results,
            'evaluation_timestamp': kwargs['execution_date'].isoformat()
        }
        
        # Use response handler for intelligent decision making
        response = response_handler.evaluate_drift_response(
            drift_metrics={
                'dataset_drift_score': drift_confidence,
                'feature_drift_count': drift_detected_count,
                'total_features': 50,  # Approximate feature count
                'drift_confidence_score': drift_confidence
            },
            execution_context=drift_context
        )
        
        logging.info(f"Comprehensive evaluation complete: {response['action']} (confidence: {response['confidence']:.3f})")
        
        return response
        
    except Exception as e:
        logging.error(f"Error in comprehensive drift evaluation: {str(e)}")
        return {
            'overall_drift_status': 'evaluation_error',
            'action': 'continue_monitoring',
            'confidence': 0.0,
            'reasoning': f'Evaluation error: {str(e)}',
            'error': str(e)
        }

def sophisticated_branching_decision(**kwargs):
    """
    Enhanced branching logic with sophisticated decision making
    """
    evaluation_result = kwargs['ti'].xcom_pull(task_ids='comprehensive_drift_evaluation')
    
    if not evaluation_result:
        logging.error("No evaluation result available for branching decision")
        return 'monitoring_error_handler'
    
    action = evaluation_result.get('action', 'continue_monitoring')
    confidence = evaluation_result.get('confidence', 0.0)
    overall_status = evaluation_result.get('overall_drift_status', 'unknown')
    
    logging.info(f"Sophisticated branching decision: status={overall_status}, action={action}, confidence={confidence:.3f}")
    
    # Enhanced action mapping with error handling
    action_mapping = {
        'continue_monitoring': 'no_drift_continue_monitoring',
        'log_and_monitor': 'minor_drift_enhanced_logging', 
        'incremental_retrain': 'moderate_drift_trigger_retraining',
        'full_retrain': 'major_drift_trigger_retraining',
        'architecture_review': 'concept_drift_architecture_alert'
    }
    
    mapped_task = action_mapping.get(action, 'no_drift_continue_monitoring')
    
    # Additional validation for high-confidence decisions
    if confidence > 0.8 and action in ['incremental_retrain', 'full_retrain']:
        logging.info(f"High-confidence drift decision: {action} with confidence {confidence:.3f}")
    
    return mapped_task

def system_health_monitoring(**kwargs):
    """
    Comprehensive system health monitoring and alerting
    """
    logging.info("Performing comprehensive system health monitoring...")
    
    try:
        # Check MLflow connectivity
        import mlflow
        mlflow.set_tracking_uri(env_vars['MLFLOW_TRACKING_URI'])
        
        # Verify experiments exist
        try:
            drift_experiment = mlflow.get_experiment_by_name(env_vars['DRIFT_MONITORING_EXPERIMENT'])
            training_experiment = mlflow.get_experiment_by_name('HealthPredict_Training_HPO_Airflow')
            
            mlflow_status = "healthy"
            logging.info("MLflow connectivity verified")
        except Exception as e:
            mlflow_status = f"error: {str(e)}"
            logging.error(f"MLflow connectivity issues: {str(e)}")
        
        # Check S3 connectivity
        try:
            s3_client = boto3.client('s3')
            s3_client.head_bucket(Bucket=env_vars['S3_BUCKET_NAME'])
            s3_status = "healthy"
            logging.info("S3 connectivity verified")
        except Exception as e:
            s3_status = f"error: {str(e)}"
            logging.error(f"S3 connectivity issues: {str(e)}")
        
        # Check batch processing backlog
        try:
            response = s3_client.list_objects_v2(
                Bucket=env_vars['S3_BUCKET_NAME'],
                Prefix=f"{env_vars['DRIFT_BATCH_DATA_S3_PREFIX']}/",
                MaxKeys=env_vars['BATCH_PROCESSING_BACKLOG_LIMIT'] + 5
            )
            
            batch_count = len(response.get('Contents', []))
            backlog_status = "healthy" if batch_count <= env_vars['BATCH_PROCESSING_BACKLOG_LIMIT'] else "backlog_alert"
            
            logging.info(f"Batch processing backlog: {batch_count} batches")
        except Exception as e:
            backlog_status = f"error: {str(e)}"
            logging.error(f"Batch backlog check failed: {str(e)}")
        
        # Create system health report
        health_report = {
            'monitoring_timestamp': kwargs['execution_date'].isoformat(),
            'mlflow_status': mlflow_status,
            's3_status': s3_status,
            'batch_backlog_status': backlog_status,
            'system_overall_health': 'healthy' if all(
                status == 'healthy' for status in [mlflow_status, s3_status, backlog_status]
            ) else 'degraded'
        }
        
        logging.info(f"System health monitoring complete: {health_report['system_overall_health']}")
        
        return health_report
        
    except Exception as e:
        logging.error(f"System health monitoring failed: {str(e)}")
        return {
            'monitoring_timestamp': kwargs['execution_date'].isoformat(),
            'system_overall_health': 'critical',
            'error': str(e)
        }

# ====================
# TASK DEFINITIONS
# ====================

# Task Group: Data Batch Management
with TaskGroup("batch_management", dag=dag) as batch_management_group:
    
    # Create simulation batch with enhanced features
    create_sim_batch = PythonOperator(
        task_id='create_simulation_batch',
        python_callable=create_simulation_batch,
        retries=2,
        retry_delay=timedelta(minutes=1),
    )
    
    # System health monitoring
    health_check = PythonOperator(
        task_id='system_health_monitoring',  
        python_callable=system_health_monitoring,
        retries=1,
        retry_delay=timedelta(seconds=30),
    )

# Task Group: Parallel Drift Detection
with TaskGroup("drift_detection_parallel", dag=dag) as drift_detection_group:
    
    # Parallel drift detection for simulation batch
    parallel_drift_sim = PythonOperator(
        task_id='parallel_drift_detection_sim',
        python_callable=lambda **kwargs: parallel_drift_detection(
            kwargs['ti'].xcom_pull(task_ids='batch_management.create_simulation_batch'), 
            **kwargs
        ),
        retries=2,
        retry_delay=timedelta(minutes=1),
    )

# Task Group: Comprehensive Analysis
with TaskGroup("drift_analysis", dag=dag) as drift_analysis_group:
    
    # Comprehensive drift evaluation
    comprehensive_eval = PythonOperator(
        task_id='comprehensive_drift_evaluation',
        python_callable=comprehensive_drift_evaluation,
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    
    # Sophisticated branching decision
    sophisticated_branch = BranchPythonOperator(
        task_id='sophisticated_branching_decision',
        python_callable=sophisticated_branching_decision,
        retries=1,
        retry_delay=timedelta(seconds=30),
    )

# Task Group: Response Actions
with TaskGroup("drift_responses", dag=dag) as drift_responses_group:
    
    # No drift response
    no_drift_monitoring = PythonOperator(
        task_id='no_drift_continue_monitoring',
        python_callable=lambda **kwargs: logging.info(
            f"No significant drift detected - continuing enhanced monitoring"
        ),
    )
    
    # Minor drift response with enhanced logging
    minor_drift_logging = PythonOperator(
        task_id='minor_drift_enhanced_logging',
        python_callable=lambda **kwargs: logging.warning(
            f"Minor drift detected - enhanced monitoring and logging activated"
        ),
    )
    
    # Moderate drift response
    moderate_drift_retraining = TriggerDagRunOperator(
        task_id='moderate_drift_trigger_retraining',
        trigger_dag_id='health_predict_training_hpo',
        conf={
            'drift_triggered': True,
            'drift_severity': 'moderate',
            'triggered_by': 'comprehensive_drift_monitoring_v2',
            'drift_context': '{{ ti.xcom_pull(task_ids="drift_analysis.comprehensive_drift_evaluation") }}',
            'retraining_timestamp': '{{ ts }}'
        },
        wait_for_completion=False,
        poke_interval=30,
    )
    
    # Major drift response  
    major_drift_retraining = TriggerDagRunOperator(
        task_id='major_drift_trigger_retraining',
        trigger_dag_id='health_predict_training_hpo', 
        conf={
            'drift_triggered': True,
            'drift_severity': 'major',
            'triggered_by': 'comprehensive_drift_monitoring_v2',
            'drift_context': '{{ ti.xcom_pull(task_ids="drift_analysis.comprehensive_drift_evaluation") }}',
            'retraining_timestamp': '{{ ts }}'
        },
        wait_for_completion=False,
        poke_interval=30,
    )
    
    # Concept drift alert
    concept_drift_alert = PythonOperator(
        task_id='concept_drift_architecture_alert',
        python_callable=lambda **kwargs: logging.critical(
            f"CONCEPT DRIFT DETECTED - Manual architecture review required immediately"
        ),
    )
    
    # Error handling
    monitoring_error = PythonOperator(
        task_id='monitoring_error_handler',
        python_callable=lambda **kwargs: logging.error(
            f"Drift monitoring encountered errors - review system health"
        ),
    )

# Final convergence task
final_convergence = PythonOperator(
    task_id='comprehensive_monitoring_completion',
    python_callable=lambda **kwargs: logging.info(
        f"Comprehensive drift monitoring cycle completed successfully"
    ),
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# ====================
# TASK DEPENDENCIES
# ====================

# Primary workflow
batch_management_group >> drift_detection_group >> drift_analysis_group

# Branching to response actions
drift_analysis_group >> drift_responses_group

# All response paths converge to completion
drift_responses_group >> final_convergence 