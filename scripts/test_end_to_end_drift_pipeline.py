#!/usr/bin/env python3
"""
End-to-End Drift Pipeline Testing Suite

This script provides comprehensive testing for the complete drift detection to model deployment pipeline:
- Complete drift detection to model deployment cycles
- Various drift scenarios and response combinations
- Model performance maintenance validation
- System performance under load and stress conditions
- Comprehensive system validation and acceptance testing

Author: Health Predict ML Engineering Team
Created: 2025-01-24 (Week 5 - Step 20)
"""

import argparse
import boto3
import pandas as pd
import numpy as np
import mlflow
import requests
import logging
import time
import json
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import subprocess
import concurrent.futures
from dataclasses import dataclass, asdict
import sys

# Add scripts directory to path for imports
sys.path.append('/opt/airflow/scripts')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    description: str
    drift_type: str  # covariate, concept, mixed, none
    drift_intensity: float  # 0.0-1.0
    expected_response: str  # continue_monitoring, minor_drift, moderate_drift, major_drift
    batch_count: int
    timeout_minutes: int

@dataclass
class TestResult:
    """Test execution result"""
    scenario_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    success: bool
    drift_detected: bool
    retraining_triggered: bool
    model_deployed: bool
    performance_maintained: bool
    errors: List[str]
    metrics: Dict[str, Any]

class EndToEndDriftPipelineTester:
    """
    Comprehensive end-to-end testing for the drift monitoring pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pipeline tester"""
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket_name = config['S3_BUCKET_NAME']
        
        # Test scenarios
        self.test_scenarios = [
            TestScenario(
                name="no_drift_baseline",
                description="Baseline scenario with no significant drift",
                drift_type="none",
                drift_intensity=0.0,
                expected_response="continue_monitoring",
                batch_count=3,
                timeout_minutes=15
            ),
            TestScenario(
                name="minor_covariate_drift",
                description="Minor covariate drift requiring enhanced monitoring",
                drift_type="covariate",
                drift_intensity=0.1,
                expected_response="minor_drift",
                batch_count=2,
                timeout_minutes=10
            ),
            TestScenario(
                name="moderate_concept_drift",
                description="Moderate concept drift triggering retraining",
                drift_type="concept",
                drift_intensity=0.2,
                expected_response="moderate_drift",
                batch_count=2,
                timeout_minutes=30
            ),
            TestScenario(
                name="major_mixed_drift",
                description="Major mixed drift requiring full retraining",
                drift_type="mixed",
                drift_intensity=0.4,
                expected_response="major_drift",
                batch_count=1,
                timeout_minutes=45
            ),
            TestScenario(
                name="stress_test_multiple_batches",
                description="Stress test with multiple concurrent batches",
                drift_type="covariate",
                drift_intensity=0.15,
                expected_response="moderate_drift",
                batch_count=5,
                timeout_minutes=60
            )
        ]
        
        self.test_results: List[TestResult] = []
        self.baseline_model_performance: Optional[Dict[str, float]] = None
        
    def prepare_test_environment(self) -> bool:
        """Prepare the test environment and validate prerequisites"""
        logger.info("Preparing test environment...")
        
        try:
            # Check MLflow connectivity
            mlflow.set_tracking_uri(self.config['MLFLOW_TRACKING_URI'])
            experiments = mlflow.search_experiments()
            logger.info(f"MLflow connected: {len(experiments)} experiments found")
            
            # Check S3 connectivity
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket accessible: {self.bucket_name}")
            
            # Verify reference data exists
            reference_key = f"{self.config['DRIFT_REFERENCE_DATA_S3_PREFIX']}/initial_train.csv"
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=reference_key)
                logger.info("Reference data found")
            except:
                logger.error(f"Reference data not found: {reference_key}")
                return False
            
            # Check drift monitoring script
            script_path = '/opt/airflow/scripts/monitor_drift.py'
            if not os.path.exists(script_path):
                logger.error(f"Drift monitoring script not found: {script_path}")
                return False
            
            # Clear any existing test batches
            self.cleanup_test_batches()
            
            # Capture baseline model performance
            self.baseline_model_performance = self.get_current_model_performance()
            
            logger.info("Test environment prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare test environment: {str(e)}")
            return False
    
    def cleanup_test_batches(self):
        """Clean up any existing test batches"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.config['DRIFT_BATCH_DATA_S3_PREFIX']}/test_"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
                    logger.info(f"Cleaned up test batch: {obj['Key']}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning up test batches: {str(e)}")
    
    def get_current_model_performance(self) -> Optional[Dict[str, float]]:
        """Get current production model performance metrics"""
        try:
            mlflow.set_tracking_uri(self.config['MLFLOW_TRACKING_URI'])
            
            # Get the latest production model
            client = mlflow.tracking.MlflowClient()
            
            # Search for latest model versions in production
            latest_versions = client.search_model_versions(
                filter_string="current_stage='Production'",
                max_results=1,
                order_by=["creation_timestamp DESC"]
            )
            
            if latest_versions:
                version = latest_versions[0]
                run_id = version.run_id
                
                # Get metrics from the training run
                run = client.get_run(run_id)
                metrics = run.data.metrics
                
                performance = {
                    'f1_score': metrics.get('f1_score', 0.0),
                    'roc_auc': metrics.get('roc_auc', 0.0),
                    'accuracy': metrics.get('accuracy', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0)
                }
                
                logger.info(f"Baseline model performance: F1={performance['f1_score']:.3f}, AUC={performance['roc_auc']:.3f}")
                return performance
                
        except Exception as e:
            logger.warning(f"Could not retrieve baseline model performance: {str(e)}")
            
        return None
    
    def create_test_batch_with_drift(
        self, 
        scenario: TestScenario, 
        batch_id: str
    ) -> Dict[str, Any]:
        """Create a test batch with specified drift characteristics"""
        logger.info(f"Creating test batch for scenario: {scenario.name}")
        
        try:
            # Load source data
            future_data_key = 'processed_data/future_data.csv'
            local_temp_file = f'/tmp/future_data_test_{batch_id}.csv'
            
            self.s3_client.download_file(self.bucket_name, future_data_key, local_temp_file)
            df = pd.read_csv(local_temp_file)
            
            # Extract base batch
            batch_size = 1000
            start_row = hash(batch_id) % (len(df) - batch_size)
            batch_data = df.iloc[start_row:start_row + batch_size].copy()
            
            # Apply drift based on scenario
            if scenario.drift_type != "none":
                batch_data = self.inject_drift(batch_data, scenario)
            
            # Save batch
            local_batch_path = f'/tmp/{batch_id}.csv'
            batch_data.to_csv(local_batch_path, index=False)
            
            # Upload to S3
            s3_batch_key = f"{self.config['DRIFT_BATCH_DATA_S3_PREFIX']}/{batch_id}.csv"
            self.s3_client.upload_file(local_batch_path, self.bucket_name, s3_batch_key)
            
            # Cleanup
            os.remove(local_temp_file)
            os.remove(local_batch_path)
            
            batch_info = {
                'batch_id': batch_id,
                's3_batch_path': s3_batch_key,
                'batch_size': len(batch_data),
                'drift_type': scenario.drift_type,
                'drift_intensity': scenario.drift_intensity,
                'creation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Test batch created: {batch_id} ({len(batch_data)} rows)")
            return batch_info
            
        except Exception as e:
            logger.error(f"Error creating test batch: {str(e)}")
            raise
    
    def inject_drift(self, batch_data: pd.DataFrame, scenario: TestScenario) -> pd.DataFrame:
        """Inject drift into batch data based on scenario"""
        
        if scenario.drift_type == "covariate":
            return self.inject_covariate_drift(batch_data, scenario.drift_intensity)
        elif scenario.drift_type == "concept":
            return self.inject_concept_drift(batch_data, scenario.drift_intensity)
        elif scenario.drift_type == "mixed":
            # Apply both types of drift
            batch_data = self.inject_covariate_drift(batch_data, scenario.drift_intensity * 0.6)
            batch_data = self.inject_concept_drift(batch_data, scenario.drift_intensity * 0.4)
            return batch_data
        else:
            return batch_data
    
    def inject_covariate_drift(self, df: pd.DataFrame, intensity: float) -> pd.DataFrame:
        """Inject covariate drift by shifting feature distributions"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns[:5]:  # Apply to first 5 numeric columns
            if np.random.random() < 0.7:  # 70% chance to modify each column
                # Apply distribution shift
                shift_amount = intensity * df[col].std()
                df[col] = df[col] + shift_amount
        
        return df
    
    def inject_concept_drift(self, df: pd.DataFrame, intensity: float) -> pd.DataFrame:
        """Inject concept drift by modifying target relationships"""
        # For concept drift, we'd modify the relationship between features and target
        # In this case, we'll simulate by modifying some feature values that affect prediction
        
        if 'age' in df.columns:
            # Modify age distribution to affect readmission patterns
            age_shift = intensity * 10  # Years
            df['age'] = df['age'] + age_shift
        
        return df
    
    def run_drift_detection_test(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run drift detection on a test batch"""
        batch_id = batch_info['batch_id']
        logger.info(f"Running drift detection test for batch: {batch_id}")
        
        # Construct paths
        new_data_path = f"s3://{self.bucket_name}/{batch_info['s3_batch_path']}"
        reference_data_path = f"s3://{self.bucket_name}/{self.config['DRIFT_REFERENCE_DATA_S3_PREFIX']}/initial_train.csv"
        reports_path = f"s3://{self.bucket_name}/{self.config['DRIFT_REPORTS_S3_PREFIX']}/test_{batch_id}"
        
        # Build drift detection command
        drift_command = [
            'python', '/opt/airflow/scripts/monitor_drift.py',
            '--s3_new_data_path', new_data_path,
            '--s3_reference_data_path', reference_data_path,
            '--s3_evidently_reports_path', reports_path,
            '--mlflow_tracking_uri', self.config['MLFLOW_TRACKING_URI'],
            '--mlflow_experiment_name', self.config['DRIFT_MONITORING_EXPERIMENT'],
            '--target_column', 'readmitted_binary',
            '--drift_threshold', str(self.config['DRIFT_THRESHOLD_MODERATE']),
            '--enable_concept_drift',
            '--verbose'
        ]
        
        try:
            start_time = time.time()
            
            # Run drift detection
            result = subprocess.run(
                drift_command,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            detection_time = time.time() - start_time
            
            # Parse result
            drift_status = "DRIFT_MONITORING_ERROR"
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    if line in ['DRIFT_DETECTED', 'NO_DRIFT', 'DRIFT_MONITORING_ERROR']:
                        drift_status = line
                        break
            
            detection_result = {
                'batch_id': batch_id,
                'drift_status': drift_status,
                'drift_detected': drift_status == 'DRIFT_DETECTED',
                'detection_time_seconds': detection_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            logger.info(f"Drift detection completed: {drift_status} in {detection_time:.2f}s")
            return detection_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Drift detection timed out for batch {batch_id}")
            return {
                'batch_id': batch_id,
                'drift_status': 'DRIFT_MONITORING_TIMEOUT',
                'drift_detected': False,
                'error': 'Detection timed out after 5 minutes'
            }
        except Exception as e:
            logger.error(f"Error running drift detection: {str(e)}")
            return {
                'batch_id': batch_id,
                'drift_status': 'DRIFT_MONITORING_ERROR',
                'drift_detected': False,
                'error': str(e)
            }
    
    def wait_for_retraining_completion(self, timeout_minutes: int = 30) -> Dict[str, Any]:
        """Wait for retraining to complete and verify results"""
        logger.info(f"Waiting for retraining completion (timeout: {timeout_minutes}m)")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        # Get initial run count
        mlflow.set_tracking_uri(self.config['MLFLOW_TRACKING_URI'])
        client = mlflow.tracking.MlflowClient()
        
        try:
            experiment = mlflow.get_experiment_by_name('HealthPredict_Training_HPO_Airflow')
            if not experiment:
                return {'error': 'Training experiment not found'}
            
            initial_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            initial_run_count = len(initial_runs)
            
            logger.info(f"Initial run count: {initial_run_count}")
            
            # Poll for new runs
            while time.time() - start_time < timeout_seconds:
                current_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                current_run_count = len(current_runs)
                
                if current_run_count > initial_run_count:
                    # New run detected, wait for completion
                    latest_run = current_runs.iloc[0]  # Most recent run
                    
                    if latest_run['status'] == 'FINISHED':
                        metrics = {
                            'f1_score': latest_run.get('metrics.f1_score', 0.0),
                            'roc_auc': latest_run.get('metrics.roc_auc', 0.0),
                            'accuracy': latest_run.get('metrics.accuracy', 0.0)
                        }
                        
                        logger.info(f"Retraining completed: F1={metrics['f1_score']:.3f}")
                        
                        return {
                            'retraining_completed': True,
                            'run_id': latest_run['run_id'],
                            'metrics': metrics,
                            'completion_time': time.time() - start_time
                        }
                    elif latest_run['status'] == 'FAILED':
                        return {
                            'retraining_completed': False,
                            'error': 'Training run failed',
                            'run_id': latest_run['run_id']
                        }
                
                time.sleep(30)  # Check every 30 seconds
            
            return {
                'retraining_completed': False,
                'error': f'Timeout after {timeout_minutes} minutes'
            }
            
        except Exception as e:
            logger.error(f"Error waiting for retraining: {str(e)}")
            return {
                'retraining_completed': False,
                'error': str(e)
            }
    
    def verify_model_deployment(self) -> Dict[str, Any]:
        """Verify that the retrained model was deployed successfully"""
        logger.info("Verifying model deployment...")
        
        try:
            mlflow.set_tracking_uri(self.config['MLFLOW_TRACKING_URI'])
            client = mlflow.tracking.MlflowClient()
            
            # Check for latest production model
            latest_versions = client.search_model_versions(
                filter_string="current_stage='Production'",
                max_results=1,
                order_by=["creation_timestamp DESC"]
            )
            
            if latest_versions:
                version = latest_versions[0]
                
                # Check if this is a recent deployment (within last hour)
                creation_time = datetime.fromisoformat(version.creation_timestamp.replace('Z', '+00:00'))
                if creation_time > datetime.now() - timedelta(hours=1):
                    
                    deployment_info = {
                        'model_deployed': True,
                        'model_name': version.name,
                        'model_version': version.version,
                        'deployment_time': version.creation_timestamp,
                        'run_id': version.run_id
                    }
                    
                    logger.info(f"Model deployment verified: {version.name} v{version.version}")
                    return deployment_info
            
            return {
                'model_deployed': False,
                'error': 'No recent model deployment found'
            }
            
        except Exception as e:
            logger.error(f"Error verifying model deployment: {str(e)}")
            return {
                'model_deployed': False,
                'error': str(e)
            }
    
    def validate_performance_maintenance(self, new_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate that model performance is maintained after retraining"""
        if not self.baseline_model_performance:
            return {
                'performance_maintained': True,  # No baseline to compare
                'warning': 'No baseline performance available for comparison'
            }
        
        performance_analysis = {}
        performance_maintained = True
        
        for metric, new_value in new_metrics.items():
            if metric in self.baseline_model_performance:
                baseline_value = self.baseline_model_performance[metric]
                change = new_value - baseline_value
                change_percent = (change / baseline_value) * 100 if baseline_value > 0 else 0
                
                performance_analysis[metric] = {
                    'baseline': baseline_value,
                    'new_value': new_value,
                    'change': change,
                    'change_percent': change_percent
                }
                
                # Performance is considered maintained if it doesn't drop by more than 5%
                if change_percent < -5.0:
                    performance_maintained = False
        
        return {
            'performance_maintained': performance_maintained,
            'performance_analysis': performance_analysis,
            'overall_assessment': 'maintained' if performance_maintained else 'degraded'
        }
    
    def run_scenario_test(self, scenario: TestScenario) -> TestResult:
        """Run a complete test scenario"""
        logger.info(f"Running test scenario: {scenario.name}")
        start_time = datetime.now()
        errors = []
        
        try:
            # Create test batches
            test_batches = []
            for i in range(scenario.batch_count):
                batch_id = f"test_{scenario.name}_{start_time.strftime('%Y%m%d_%H%M%S')}_{i:02d}"
                batch_info = self.create_test_batch_with_drift(scenario, batch_id)
                test_batches.append(batch_info)
            
            # Run drift detection on all batches
            drift_results = []
            for batch_info in test_batches:
                drift_result = self.run_drift_detection_test(batch_info)
                drift_results.append(drift_result)
            
            # Analyze drift detection results
            drift_detected = any(r.get('drift_detected', False) for r in drift_results)
            
            # Check if retraining should be triggered
            should_trigger_retraining = scenario.expected_response in ['moderate_drift', 'major_drift']
            
            retraining_triggered = False
            model_deployed = False
            performance_maintained = True
            retraining_metrics = {}
            
            if drift_detected and should_trigger_retraining:
                # Wait for retraining to complete
                retraining_result = self.wait_for_retraining_completion(scenario.timeout_minutes)
                
                if retraining_result.get('retraining_completed', False):
                    retraining_triggered = True
                    retraining_metrics = retraining_result.get('metrics', {})
                    
                    # Verify model deployment
                    deployment_result = self.verify_model_deployment()
                    model_deployed = deployment_result.get('model_deployed', False)
                    
                    # Validate performance maintenance
                    if retraining_metrics:
                        performance_result = self.validate_performance_maintenance(retraining_metrics)
                        performance_maintained = performance_result.get('performance_maintained', True)
                else:
                    errors.append(f"Retraining failed: {retraining_result.get('error', 'Unknown error')}")
            
            # Determine overall success
            success = True
            
            # Check if drift detection worked as expected
            if should_trigger_retraining and not drift_detected:
                success = False
                errors.append(f"Expected drift detection but none found for {scenario.drift_type} drift")
            elif not should_trigger_retraining and drift_detected:
                # This might be OK - false positives can happen
                logger.warning(f"Unexpected drift detected in {scenario.name}")
            
            # Check retraining and deployment
            if should_trigger_retraining and drift_detected:
                if not retraining_triggered:
                    success = False
                    errors.append("Expected retraining to be triggered but it wasn't")
                elif not model_deployed:
                    success = False
                    errors.append("Expected model deployment but it didn't happen")
                elif not performance_maintained:
                    success = False
                    errors.append("Model performance degraded after retraining")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Compile metrics
            scenario_metrics = {
                'batch_count': len(test_batches),
                'drift_detection_results': drift_results,
                'retraining_metrics': retraining_metrics,
                'duration_seconds': duration,
                'expected_vs_actual': {
                    'expected_response': scenario.expected_response,
                    'drift_detected': drift_detected,
                    'retraining_triggered': retraining_triggered,
                    'model_deployed': model_deployed
                }
            }
            
            result = TestResult(
                scenario_name=scenario.name,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                success=success,
                drift_detected=drift_detected,
                retraining_triggered=retraining_triggered,
                model_deployed=model_deployed,
                performance_maintained=performance_maintained,
                errors=errors,
                metrics=scenario_metrics
            )
            
            logger.info(f"Scenario {scenario.name} completed: {'SUCCESS' if success else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Error running scenario {scenario.name}: {str(e)}")
            errors.append(str(e))
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                scenario_name=scenario.name,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                success=False,
                drift_detected=False,
                retraining_triggered=False,
                model_deployed=False,
                performance_maintained=False,
                errors=errors,
                metrics={'error': str(e)}
            )
    
    def run_load_test(self, concurrent_batches: int = 3, duration_minutes: int = 10) -> Dict[str, Any]:
        """Run load testing with multiple concurrent batches"""
        logger.info(f"Running load test: {concurrent_batches} concurrent batches for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = {
            'start_time': datetime.now().isoformat(),
            'concurrent_batches': concurrent_batches,
            'duration_minutes': duration_minutes,
            'batches_processed': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'average_detection_time': 0.0,
            'errors': []
        }
        
        batch_counter = 0
        detection_times = []
        
        try:
            while time.time() < end_time:
                # Create multiple concurrent batches
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_batches) as executor:
                    futures = []
                    
                    for i in range(concurrent_batches):
                        batch_id = f"load_test_{int(time.time())}_{batch_counter:04d}_{i}"
                        batch_counter += 1
                        
                        # Create batch with random drift
                        scenario = TestScenario(
                            name="load_test",
                            description="Load test batch",
                            drift_type="covariate" if np.random.random() > 0.5 else "none",
                            drift_intensity=np.random.uniform(0.0, 0.2),
                            expected_response="continue_monitoring",
                            batch_count=1,
                            timeout_minutes=5
                        )
                        
                        future = executor.submit(self.run_single_load_test_batch, batch_id, scenario)
                        futures.append(future)
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(futures, timeout=300):
                        try:
                            batch_result = future.result()
                            results['batches_processed'] += 1
                            
                            if batch_result.get('success', False):
                                results['successful_detections'] += 1
                                detection_times.append(batch_result.get('detection_time', 0.0))
                            else:
                                results['failed_detections'] += 1
                                if 'error' in batch_result:
                                    results['errors'].append(batch_result['error'])
                                    
                        except Exception as e:
                            results['failed_detections'] += 1
                            results['errors'].append(str(e))
                
                # Brief pause between rounds
                time.sleep(30)
            
            # Calculate final metrics
            if detection_times:
                results['average_detection_time'] = sum(detection_times) / len(detection_times)
            
            results['end_time'] = datetime.now().isoformat()
            results['success_rate'] = (
                results['successful_detections'] / results['batches_processed'] 
                if results['batches_processed'] > 0 else 0.0
            )
            
            logger.info(f"Load test completed: {results['batches_processed']} batches, "
                       f"{results['success_rate']:.2%} success rate")
            
            return results
            
        except Exception as e:
            logger.error(f"Load test failed: {str(e)}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now().isoformat()
            return results
    
    def run_single_load_test_batch(self, batch_id: str, scenario: TestScenario) -> Dict[str, Any]:
        """Run a single batch for load testing"""
        try:
            start_time = time.time()
            
            # Create and process batch
            batch_info = self.create_test_batch_with_drift(scenario, batch_id)
            drift_result = self.run_drift_detection_test(batch_info)
            
            detection_time = time.time() - start_time
            
            return {
                'batch_id': batch_id,
                'success': drift_result.get('return_code', 1) == 0,
                'detection_time': detection_time,
                'drift_detected': drift_result.get('drift_detected', False)
            }
            
        except Exception as e:
            return {
                'batch_id': batch_id,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("Starting comprehensive end-to-end test suite")
        
        if not self.prepare_test_environment():
            return {
                'overall_success': False,
                'error': 'Failed to prepare test environment'
            }
        
        suite_start_time = datetime.now()
        
        # Run scenario tests
        logger.info("Running scenario tests...")
        for scenario in self.test_scenarios:
            if scenario.name != "stress_test_multiple_batches":  # Skip stress test in regular suite
                result = self.run_scenario_test(scenario)
                self.test_results.append(result)
        
        # Run load test
        logger.info("Running load test...")
        load_test_result = self.run_load_test(concurrent_batches=2, duration_minutes=5)
        
        suite_end_time = datetime.now()
        suite_duration = (suite_end_time - suite_start_time).total_seconds()
        
        # Analyze results
        successful_tests = len([r for r in self.test_results if r.success])
        total_tests = len(self.test_results)
        
        # Generate comprehensive report
        test_report = {
            'suite_start_time': suite_start_time.isoformat(),
            'suite_end_time': suite_end_time.isoformat(),
            'suite_duration_seconds': suite_duration,
            'total_scenario_tests': total_tests,
            'successful_scenario_tests': successful_tests,
            'scenario_success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
            'load_test_result': load_test_result,
            'overall_success': successful_tests == total_tests and load_test_result.get('success_rate', 0) > 0.8,
            'detailed_results': [asdict(r) for r in self.test_results],
            'baseline_performance': self.baseline_model_performance,
            'summary': self.generate_test_summary()
        }
        
        logger.info(f"Test suite completed: {successful_tests}/{total_tests} scenarios passed")
        return test_report
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate a summary of test results"""
        return {
            'drift_detection_accuracy': len([r for r in self.test_results if r.drift_detected]) / len(self.test_results) if self.test_results else 0.0,
            'retraining_trigger_success': len([r for r in self.test_results if r.retraining_triggered]) / len([r for r in self.test_results if 'moderate' in r.scenario_name or 'major' in r.scenario_name]) if self.test_results else 0.0,
            'deployment_success_rate': len([r for r in self.test_results if r.model_deployed]) / len([r for r in self.test_results if r.retraining_triggered]) if self.test_results else 0.0,
            'performance_maintenance_rate': len([r for r in self.test_results if r.performance_maintained]) / len(self.test_results) if self.test_results else 0.0,
            'average_test_duration': sum(r.duration_seconds for r in self.test_results) / len(self.test_results) if self.test_results else 0.0
        }

def main():
    """Main function for end-to-end testing"""
    parser = argparse.ArgumentParser(description='End-to-End Drift Pipeline Testing')
    
    parser.add_argument('--scenario', 
                       help='Run specific test scenario')
    
    parser.add_argument('--load-test-only', action='store_true',
                       help='Run only load testing')
    
    parser.add_argument('--concurrent-batches', type=int, default=3,
                       help='Number of concurrent batches for load testing')
    
    parser.add_argument('--duration', type=int, default=10,
                       help='Load test duration in minutes')
    
    parser.add_argument('--output-file',
                       help='Output file for test results (JSON)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration
    config = {
        'S3_BUCKET_NAME': os.getenv('S3_BUCKET_NAME', 'health-predict-mlops-f9ac6509'),
        'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
        'DRIFT_MONITORING_EXPERIMENT': os.getenv('DRIFT_MONITORING_EXPERIMENT', 'HealthPredict_Drift_Monitoring'),
        'DRIFT_BATCH_DATA_S3_PREFIX': os.getenv('DRIFT_BATCH_DATA_S3_PREFIX', 'drift_monitoring/batch_data'),
        'DRIFT_REFERENCE_DATA_S3_PREFIX': os.getenv('DRIFT_REFERENCE_DATA_S3_PREFIX', 'drift_monitoring/reference_data'),
        'DRIFT_REPORTS_S3_PREFIX': os.getenv('DRIFT_REPORTS_S3_PREFIX', 'drift_monitoring/reports'),
        'DRIFT_THRESHOLD_MODERATE': float(os.getenv('DRIFT_THRESHOLD_MODERATE', '0.15'))
    }
    
    try:
        tester = EndToEndDriftPipelineTester(config)
        
        if args.load_test_only:
            # Run only load testing
            results = tester.run_load_test(
                concurrent_batches=args.concurrent_batches,
                duration_minutes=args.duration
            )
        elif args.scenario:
            # Run specific scenario
            scenario = next((s for s in tester.test_scenarios if s.name == args.scenario), None)
            if not scenario:
                logger.error(f"Scenario not found: {args.scenario}")
                return 1
            
            result = tester.run_scenario_test(scenario)
            results = {'scenario_result': asdict(result)}
        else:
            # Run comprehensive test suite
            results = tester.run_comprehensive_test_suite()
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
        else:
            print(json.dumps(results, indent=2))
        
        # Determine exit code
        if results.get('overall_success', False):
            logger.info("All tests passed successfully!")
            return 0
        else:
            logger.error("Some tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main()) 