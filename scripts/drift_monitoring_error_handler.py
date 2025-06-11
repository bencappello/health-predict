#!/usr/bin/env python3
"""
Drift Monitoring Error Handler and Recovery System

This module provides comprehensive error handling and recovery for the drift monitoring pipeline:
- Comprehensive error classification and handling
- Automatic retry mechanisms with exponential backoff
- Error notification and escalation procedures
- Graceful degradation for partial failures
- System health monitoring and recovery procedures

Author: Health Predict ML Engineering Team
Created: 2025-01-24 (Week 5 - Step 19)
"""

import logging
import time
import boto3
import mlflow
import json
import os
import smtplib
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
import functools
import subprocess
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels for classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for systematic handling"""
    DATA_ACCESS = "data_access"
    PROCESSING = "processing"
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE = "resource"
    VALIDATION = "validation"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Available recovery actions"""
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    DEGRADE_GRACEFULLY = "degrade_gracefully"
    ESCALATE = "escalate"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAK = "circuit_break"

@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    error_id: str
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function_name: str
    error_message: str
    stack_trace: str
    recovery_action: RecoveryAction
    retry_count: int
    max_retries: int
    execution_context: Dict[str, Any]
    system_state: Dict[str, Any]
    recommended_action: str

@dataclass 
class SystemHealthStatus:
    """System health monitoring status"""
    component: str
    status: str  # healthy, degraded, critical, unavailable
    response_time_ms: Optional[float]
    error_rate: float
    last_check: str
    details: Dict[str, Any]

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class DriftMonitoringErrorHandler:
    """
    Comprehensive error handling and recovery system for drift monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the error handler"""
        self.config = config
        self.circuit_breakers = {
            'mlflow': CircuitBreaker(failure_threshold=3, recovery_timeout=120),
            's3': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'drift_detection': CircuitBreaker(failure_threshold=2, recovery_timeout=180)
        }
        
        # Error classification rules
        self.error_classification_rules = {
            # Data access errors
            'NoSuchBucket': (ErrorSeverity.HIGH, ErrorCategory.DATA_ACCESS, RecoveryAction.ESCALATE),
            'NoSuchKey': (ErrorSeverity.MEDIUM, ErrorCategory.DATA_ACCESS, RecoveryAction.RETRY_WITH_BACKOFF),
            'AccessDenied': (ErrorSeverity.HIGH, ErrorCategory.CONFIGURATION, RecoveryAction.ESCALATE),
            
            # Processing errors
            'TimeoutError': (ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING, RecoveryAction.RETRY_WITH_BACKOFF),
            'MemoryError': (ErrorSeverity.HIGH, ErrorCategory.RESOURCE, RecoveryAction.DEGRADE_GRACEFULLY),
            'ConnectionError': (ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE, RecoveryAction.CIRCUIT_BREAK),
            
            # MLflow errors
            'MlflowException': (ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE, RecoveryAction.RETRY_WITH_BACKOFF),
            'RestException': (ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE, RecoveryAction.CIRCUIT_BREAK),
            
            # Infrastructure errors
            'BotoCoreError': (ErrorSeverity.HIGH, ErrorCategory.INFRASTRUCTURE, RecoveryAction.RETRY_WITH_BACKOFF),
            'EndpointConnectionError': (ErrorSeverity.HIGH, ErrorCategory.INFRASTRUCTURE, RecoveryAction.CIRCUIT_BREAK),
            
            # Validation errors
            'ValueError': (ErrorSeverity.LOW, ErrorCategory.VALIDATION, RecoveryAction.FAIL_FAST),
            'KeyError': (ErrorSeverity.LOW, ErrorCategory.VALIDATION, RecoveryAction.FAIL_FAST),
        }
        
        self.system_health_status: Dict[str, SystemHealthStatus] = {}
        self.error_history: List[ErrorContext] = []
        
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Classify error and determine appropriate recovery action"""
        
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Determine severity, category, and recovery action
        if error_type in self.error_classification_rules:
            severity, category, recovery_action = self.error_classification_rules[error_type]
        else:
            # Default classification for unknown errors
            severity = ErrorSeverity.MEDIUM
            category = ErrorCategory.UNKNOWN
            recovery_action = RecoveryAction.RETRY_WITH_BACKOFF
        
        # Enhanced classification based on error message content
        severity, category, recovery_action = self._enhance_classification(
            error_message, severity, category, recovery_action
        )
        
        # Generate error context
        error_context = ErrorContext(
            error_id=f"drift_error_{int(time.time())}_{hash(error_message) % 10000}",
            timestamp=datetime.now().isoformat(),
            severity=severity,
            category=category,
            component=context.get('component', 'unknown'),
            function_name=context.get('function_name', 'unknown'),
            error_message=error_message,
            stack_trace=stack_trace,
            recovery_action=recovery_action,
            retry_count=context.get('retry_count', 0),
            max_retries=context.get('max_retries', 3),
            execution_context=context,
            system_state=self._capture_system_state(),
            recommended_action=self._generate_recommended_action(error_type, error_message, context)
        )
        
        self.error_history.append(error_context)
        return error_context
    
    def _enhance_classification(
        self, 
        error_message: str, 
        severity: ErrorSeverity, 
        category: ErrorCategory, 
        recovery_action: RecoveryAction
    ) -> tuple:
        """Enhance error classification based on message content"""
        
        error_msg_lower = error_message.lower()
        
        # Upgrade severity for critical keywords
        if any(keyword in error_msg_lower for keyword in ['critical', 'fatal', 'emergency']):
            severity = ErrorSeverity.CRITICAL
            recovery_action = RecoveryAction.ESCALATE
        
        # Resource exhaustion patterns
        elif any(keyword in error_msg_lower for keyword in ['memory', 'disk', 'timeout', 'quota']):
            if category == ErrorCategory.UNKNOWN:
                category = ErrorCategory.RESOURCE
            if severity == ErrorSeverity.LOW:
                severity = ErrorSeverity.MEDIUM
        
        # Configuration issues
        elif any(keyword in error_msg_lower for keyword in ['config', 'permission', 'auth', 'credential']):
            category = ErrorCategory.CONFIGURATION
            recovery_action = RecoveryAction.ESCALATE
        
        return severity, category, recovery_action
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': {name: asdict(status) for name, status in self.system_health_status.items()},
                'circuit_breaker_states': {
                    name: breaker.state for name, breaker in self.circuit_breakers.items()
                },
                'recent_error_count': len([e for e in self.error_history 
                                          if datetime.fromisoformat(e.timestamp) > datetime.now() - timedelta(hours=1)]),
                'memory_usage': self._get_memory_usage(),
                'disk_usage': self._get_disk_usage()
            }
        except Exception as e:
            logger.warning(f"Failed to capture system state: {str(e)}")
            return {'error': f'Failed to capture state: {str(e)}'}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3)
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information"""
        try:
            import shutil
            usage = shutil.disk_usage('/')
            total_gb = usage.total / (1024**3)
            free_gb = usage.free / (1024**3)
            used_percent = ((usage.total - usage.free) / usage.total) * 100
            
            return {
                'total_gb': total_gb,
                'free_gb': free_gb,
                'used_percent': used_percent
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommended_action(self, error_type: str, error_message: str, context: Dict[str, Any]) -> str:
        """Generate human-readable recommended action"""
        
        if error_type == 'NoSuchBucket':
            return "Verify S3 bucket name and AWS credentials. Check bucket exists and is accessible."
        elif error_type == 'NoSuchKey':
            return "Verify S3 object key exists. Check data pipeline has created expected files."
        elif error_type == 'AccessDenied':
            return "Check IAM permissions for S3/MLflow access. Verify role policies are correct."
        elif error_type == 'TimeoutError':
            return "Check network connectivity and service availability. Consider increasing timeout values."
        elif error_type == 'MemoryError':
            return "Reduce batch size or enable data streaming. Monitor memory usage and consider scaling up."
        elif error_type == 'MlflowException':
            return "Check MLflow server connectivity and status. Verify tracking URI configuration."
        elif 'connection' in error_message.lower():
            return "Check network connectivity to external services. Verify firewall and security group settings."
        else:
            return f"Review error details and context. Consider retry if error is transient."
    
    def retry_with_exponential_backoff(
        self,
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute function with exponential backoff retry strategy"""
        
        if context is None:
            context = {}
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                context['retry_count'] = attempt
                context['max_retries'] = max_retries
                
                if attempt > 0:
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                
                result = func()
                
                if attempt > 0:
                    logger.info(f"Retry successful for {func.__name__} after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_error = e
                error_context = self.classify_error(e, context)
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                
                # Log error details
                self._log_error_details(error_context)
                
                # Check if we should continue retrying
                if attempt >= max_retries:
                    logger.error(f"All retry attempts exhausted for {func.__name__}")
                    break
                
                # Check if error suggests we shouldn't retry
                if error_context.recovery_action == RecoveryAction.FAIL_FAST:
                    logger.error(f"Fail-fast error detected, stopping retries for {func.__name__}")
                    break
        
        # All retries exhausted, escalate or fail
        if last_error:
            final_error_context = self.classify_error(last_error, context)
            self._handle_final_failure(final_error_context)
            raise last_error
    
    def _log_error_details(self, error_context: ErrorContext):
        """Log comprehensive error details"""
        logger.error(f"Error ID: {error_context.error_id}")
        logger.error(f"Severity: {error_context.severity.value}")
        logger.error(f"Category: {error_context.category.value}")
        logger.error(f"Component: {error_context.component}")
        logger.error(f"Message: {error_context.error_message}")
        logger.error(f"Recommended Action: {error_context.recommended_action}")
        
        # Log to MLflow if available
        try:
            if self.circuit_breakers['mlflow'].state != 'OPEN':
                self._log_error_to_mlflow(error_context)
        except Exception as e:
            logger.warning(f"Failed to log error to MLflow: {str(e)}")
    
    def _log_error_to_mlflow(self, error_context: ErrorContext):
        """Log error details to MLflow for tracking"""
        try:
            mlflow.set_tracking_uri(self.config.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
            
            with mlflow.start_run(run_name=f"error_log_{error_context.error_id}"):
                mlflow.log_param("error_id", error_context.error_id)
                mlflow.log_param("severity", error_context.severity.value)
                mlflow.log_param("category", error_context.category.value)
                mlflow.log_param("component", error_context.component)
                mlflow.log_param("recovery_action", error_context.recovery_action.value)
                
                mlflow.log_metric("retry_count", error_context.retry_count)
                mlflow.log_metric("max_retries", error_context.max_retries)
                
                # Log error details as artifact
                error_details = {
                    'error_context': asdict(error_context),
                    'system_state': error_context.system_state
                }
                
                with open('/tmp/error_details.json', 'w') as f:
                    json.dump(error_details, f, indent=2)
                
                mlflow.log_artifact('/tmp/error_details.json')
                os.remove('/tmp/error_details.json')
                
        except Exception as e:
            logger.warning(f"Failed to log error to MLflow: {str(e)}")
    
    def _handle_final_failure(self, error_context: ErrorContext):
        """Handle final failure after all retries exhausted"""
        
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_alert_notification(error_context)
        
        # Update system health status
        self._update_system_health(error_context.component, 'critical', error_context)
        
        # Log final failure
        logger.critical(f"FINAL FAILURE - {error_context.component}: {error_context.error_message}")
        logger.critical(f"Recommended Action: {error_context.recommended_action}")
    
    def _send_alert_notification(self, error_context: ErrorContext):
        """Send alert notification for critical errors"""
        try:
            alert_message = self._format_alert_message(error_context)
            
            # Log alert (since we don't have email/Slack configured in this environment)
            logger.critical("ALERT NOTIFICATION:")
            logger.critical(alert_message)
            
            # In a production environment, this would send to email/Slack
            # self._send_email_alert(alert_message)
            # self._send_slack_alert(alert_message)
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {str(e)}")
    
    def _format_alert_message(self, error_context: ErrorContext) -> str:
        """Format alert message for notifications"""
        return f"""
DRIFT MONITORING ALERT - {error_context.severity.value.upper()}

Error ID: {error_context.error_id}
Timestamp: {error_context.timestamp}
Component: {error_context.component}
Category: {error_context.category.value}

Error Message: {error_context.error_message}

Recommended Action: {error_context.recommended_action}

System State:
- Circuit Breakers: {error_context.system_state.get('circuit_breaker_states', {})}
- Recent Errors: {error_context.system_state.get('recent_error_count', 0)}

This requires immediate attention.
        """.strip()
    
    def monitor_system_health(self) -> Dict[str, SystemHealthStatus]:
        """Comprehensive system health monitoring"""
        logger.info("Performing system health monitoring...")
        
        components_to_check = [
            ('mlflow', self._check_mlflow_health),
            ('s3', self._check_s3_health),
            ('drift_detection', self._check_drift_detection_health),
            ('airflow', self._check_airflow_health)
        ]
        
        for component, health_check_func in components_to_check:
            try:
                start_time = time.time()
                status_details = health_check_func()
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                self.system_health_status[component] = SystemHealthStatus(
                    component=component,
                    status=status_details.get('status', 'unknown'),
                    response_time_ms=response_time,
                    error_rate=status_details.get('error_rate', 0.0),
                    last_check=datetime.now().isoformat(),
                    details=status_details
                )
                
            except Exception as e:
                logger.error(f"Health check failed for {component}: {str(e)}")
                self.system_health_status[component] = SystemHealthStatus(
                    component=component,
                    status='critical',
                    response_time_ms=None,
                    error_rate=1.0,
                    last_check=datetime.now().isoformat(),
                    details={'error': str(e)}
                )
        
        return self.system_health_status
    
    def _check_mlflow_health(self) -> Dict[str, Any]:
        """Check MLflow server health"""
        try:
            mlflow.set_tracking_uri(self.config.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
            
            # Try to list experiments
            experiments = mlflow.search_experiments()
            
            return {
                'status': 'healthy',
                'experiments_count': len(experiments),
                'tracking_uri': self.config.get('MLFLOW_TRACKING_URI')
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'error': str(e),
                'tracking_uri': self.config.get('MLFLOW_TRACKING_URI')
            }
    
    def _check_s3_health(self) -> Dict[str, Any]:
        """Check S3 connectivity and access"""
        try:
            s3_client = boto3.client('s3')
            bucket_name = self.config.get('S3_BUCKET_NAME')
            
            # Check bucket access
            s3_client.head_bucket(Bucket=bucket_name)
            
            # Check key prefixes exist
            prefixes_to_check = [
                self.config.get('DRIFT_BATCH_DATA_S3_PREFIX', 'drift_monitoring/batch_data'),
                self.config.get('DRIFT_REPORTS_S3_PREFIX', 'drift_monitoring/reports')
            ]
            
            prefix_status = {}
            for prefix in prefixes_to_check:
                try:
                    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
                    prefix_status[prefix] = 'accessible'
                except Exception as e:
                    prefix_status[prefix] = f'error: {str(e)}'
            
            return {
                'status': 'healthy',
                'bucket_name': bucket_name,
                'prefix_status': prefix_status
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'error': str(e),
                'bucket_name': self.config.get('S3_BUCKET_NAME')
            }
    
    def _check_drift_detection_health(self) -> Dict[str, Any]:
        """Check drift detection script health"""
        try:
            # Check if drift detection script exists and is executable
            script_path = '/opt/airflow/scripts/monitor_drift.py'
            
            if not os.path.exists(script_path):
                return {
                    'status': 'critical',
                    'error': f'Drift detection script not found: {script_path}'
                }
            
            # Try to run script with help flag (quick test)
            result = subprocess.run(
                ['python', script_path, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    'status': 'healthy',
                    'script_path': script_path,
                    'help_output_length': len(result.stdout)
                }
            else:
                return {
                    'status': 'degraded',
                    'error': f'Script help failed: {result.stderr}',
                    'return_code': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'degraded',
                'error': 'Script help command timed out'
            }
        except Exception as e:
            return {
                'status': 'critical',
                'error': str(e)
            }
    
    def _check_airflow_health(self) -> Dict[str, Any]:
        """Check Airflow connectivity (basic check)"""
        try:
            # Basic check - verify we're in Airflow environment
            airflow_home = os.getenv('AIRFLOW_HOME')
            
            if airflow_home and os.path.exists(airflow_home):
                return {
                    'status': 'healthy',
                    'airflow_home': airflow_home,
                    'environment': 'airflow_worker'
                }
            else:
                return {
                    'status': 'degraded',
                    'warning': 'AIRFLOW_HOME not set or not found',
                    'airflow_home': airflow_home
                }
                
        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e)
            }
    
    def _update_system_health(self, component: str, status: str, error_context: Optional[ErrorContext] = None):
        """Update system health status for a component"""
        self.system_health_status[component] = SystemHealthStatus(
            component=component,
            status=status,
            response_time_ms=None,
            error_rate=1.0 if status in ['critical', 'degraded'] else 0.0,
            last_check=datetime.now().isoformat(),
            details={'error_context': asdict(error_context) if error_context else None}
        )
    
    def implement_graceful_degradation(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Implement graceful degradation strategies"""
        logger.info(f"Implementing graceful degradation for {error_context.component}")
        
        degradation_strategies = {
            'mlflow': self._degrade_mlflow_gracefully,
            's3': self._degrade_s3_gracefully,
            'drift_detection': self._degrade_drift_detection_gracefully
        }
        
        component = error_context.component
        if component in degradation_strategies:
            return degradation_strategies[component](error_context)
        else:
            return self._default_graceful_degradation(error_context)
    
    def _degrade_mlflow_gracefully(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Graceful degradation for MLflow failures"""
        return {
            'strategy': 'local_logging',
            'description': 'Switch to local file logging when MLflow unavailable',
            'fallback_path': '/tmp/drift_monitoring_fallback_logs',
            'impact': 'Experiment tracking disabled, local logs maintained',
            'recovery_suggestion': 'Check MLflow server status and connectivity'
        }
    
    def _degrade_s3_gracefully(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Graceful degradation for S3 failures"""
        return {
            'strategy': 'local_storage',
            'description': 'Use local storage for batch data and reports',
            'fallback_path': '/tmp/drift_monitoring_s3_fallback',
            'impact': 'Reports stored locally, manual S3 sync required',
            'recovery_suggestion': 'Check S3 connectivity and IAM permissions'
        }
    
    def _degrade_drift_detection_gracefully(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Graceful degradation for drift detection failures"""
        return {
            'strategy': 'basic_monitoring',
            'description': 'Switch to basic statistical monitoring',
            'fallback_method': 'simple_variance_tracking',
            'impact': 'Reduced drift detection accuracy, basic alerts only',
            'recovery_suggestion': 'Check dependencies and data quality'
        }
    
    def _default_graceful_degradation(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Default graceful degradation strategy"""
        return {
            'strategy': 'minimal_operation',
            'description': 'Continue with minimal functionality',
            'impact': 'Reduced system capabilities, manual intervention may be required',
            'recovery_suggestion': error_context.recommended_action
        }
    
    def generate_error_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        recent_errors = [
            e for e in self.error_history 
            if datetime.fromisoformat(e.timestamp) > cutoff_time
        ]
        
        # Error statistics
        error_stats = {
            'total_errors': len(recent_errors),
            'errors_by_severity': {},
            'errors_by_category': {},
            'errors_by_component': {},
            'most_common_errors': {}
        }
        
        for error in recent_errors:
            # By severity
            severity = error.severity.value
            error_stats['errors_by_severity'][severity] = error_stats['errors_by_severity'].get(severity, 0) + 1
            
            # By category
            category = error.category.value
            error_stats['errors_by_category'][category] = error_stats['errors_by_category'].get(category, 0) + 1
            
            # By component
            component = error.component
            error_stats['errors_by_component'][component] = error_stats['errors_by_component'].get(component, 0) + 1
            
            # Most common error messages
            error_msg = error.error_message[:100]  # Truncate for grouping
            error_stats['most_common_errors'][error_msg] = error_stats['most_common_errors'].get(error_msg, 0) + 1
        
        # System health summary
        health_summary = {
            component: {
                'status': status.status,
                'last_check': status.last_check,
                'response_time_ms': status.response_time_ms
            }
            for component, status in self.system_health_status.items()
        }
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'time_range_hours': time_range_hours,
            'error_statistics': error_stats,
            'system_health_summary': health_summary,
            'circuit_breaker_states': {
                name: breaker.state for name, breaker in self.circuit_breakers.items()
            },
            'recent_critical_errors': [
                asdict(e) for e in recent_errors 
                if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            ][:10]  # Last 10 critical errors
        }

# Decorators for easy error handling

def with_error_handling(error_handler: DriftMonitoringErrorHandler, 
                       component: str = "unknown",
                       max_retries: int = 3):
    """Decorator to add comprehensive error handling to functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = {
                'component': component,
                'function_name': func.__name__,
                'args': str(args)[:200],  # Truncate for logging
                'kwargs': str(kwargs)[:200]
            }
            
            def wrapped_func():
                return func(*args, **kwargs)
            
            try:
                return error_handler.retry_with_exponential_backoff(
                    wrapped_func,
                    max_retries=max_retries,
                    context=context
                )
            except Exception as e:
                error_context = error_handler.classify_error(e, context)
                if error_context.recovery_action == RecoveryAction.DEGRADE_GRACEFULLY:
                    degradation_info = error_handler.implement_graceful_degradation(error_context)
                    logger.warning(f"Graceful degradation activated: {degradation_info['strategy']}")
                raise
        
        return wrapper
    return decorator

def with_circuit_breaker(error_handler: DriftMonitoringErrorHandler, service: str):
    """Decorator to add circuit breaker protection"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            circuit_breaker = error_handler.circuit_breakers.get(service)
            if circuit_breaker:
                return circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator 