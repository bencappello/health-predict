#!/usr/bin/env python3
"""
Graduated Drift Response System for Health Predict MLOps Pipeline

This module implements intelligent drift response logic based on severity levels:
- Minor drift: logging and monitoring
- Moderate drift: incremental retraining  
- Major drift: full retraining
- Concept drift: architecture review alerts

Usage:
    from scripts.drift_response_handler import DriftResponseHandler
    
    handler = DriftResponseHandler()
    response = handler.evaluate_drift_response(drift_metrics, drift_context)
"""

import logging
import json
import time
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import boto3
from airflow.models import Variable

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DriftSeverity(Enum):
    """Enumeration of drift severity levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONCEPT = "concept"

class ResponseAction(Enum):
    """Enumeration of possible response actions"""
    CONTINUE_MONITORING = "continue_monitoring"
    LOG_AND_MONITOR = "log_and_monitor"
    INCREMENTAL_RETRAIN = "incremental_retrain"
    FULL_RETRAIN = "full_retrain"
    ARCHITECTURE_REVIEW = "architecture_review"
    EMERGENCY_ALERT = "emergency_alert"

@dataclass
class DriftMetrics:
    """Data class for drift detection metrics"""
    dataset_drift_score: float
    feature_drift_count: int
    total_features: int
    concept_drift_score: Optional[float] = None
    prediction_drift_score: Optional[float] = None
    performance_degradation: Optional[float] = None
    confidence_score: Optional[float] = None
    drift_methods: Optional[Dict[str, float]] = None

@dataclass
class DriftResponse:
    """Data class for drift response recommendations"""
    severity: DriftSeverity
    action: ResponseAction
    confidence: float
    reasoning: str
    recommendations: List[str]
    escalation_needed: bool
    metadata: Dict

class DriftResponseHandler:
    """
    Main class for handling graduated drift response based on severity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize drift response handler with configuration
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_default_config()
        self.response_history = []
        
        # Initialize AWS clients if needed
        try:
            self.s3_client = boto3.client('s3')
        except Exception as e:
            logger.warning(f"Could not initialize S3 client: {e}")
            self.s3_client = None
            
        logger.info("DriftResponseHandler initialized")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration for drift response thresholds"""
        return {
            # Drift severity thresholds
            'minor_drift_threshold': 0.05,
            'moderate_drift_threshold': 0.15,
            'major_drift_threshold': 0.30,
            'concept_drift_threshold': 0.20,
            
            # Feature drift thresholds (percentage of features showing drift)
            'minor_feature_drift_pct': 0.10,  # 10% of features
            'moderate_feature_drift_pct': 0.25,  # 25% of features
            'major_feature_drift_pct': 0.50,  # 50% of features
            
            # Performance degradation thresholds
            'minor_performance_drop': 0.02,  # 2% F1 score drop
            'moderate_performance_drop': 0.05,  # 5% F1 score drop
            'major_performance_drop': 0.10,  # 10% F1 score drop
            
            # Response action settings
            'enable_automatic_retraining': True,
            'retraining_cooldown_hours': 6,  # Minimum time between retraining events
            'escalation_threshold': 3,  # Number of consecutive major drifts before escalation
            
            # Alert settings  
            'enable_alerts': True,
            'alert_methods': ['logging'],  # Can include 'email', 'slack'
            
            # S3 paths for logging
            's3_bucket': 'health-predict-mlops-f9ac6509',
            'drift_response_log_prefix': 'drift_monitoring/response_logs/',
        }
    
    def evaluate_drift_response(
        self, 
        drift_metrics: Union[DriftMetrics, Dict], 
        drift_context: Optional[Dict] = None
    ) -> DriftResponse:
        """
        Evaluate drift metrics and determine appropriate response
        
        Args:
            drift_metrics: DriftMetrics object or dictionary with drift metrics
            drift_context: Optional context information about the drift detection
            
        Returns:
            DriftResponse object with recommended action and metadata
        """
        logger.info("Evaluating drift response...")
        
        # Convert dict to DriftMetrics if needed
        if isinstance(drift_metrics, dict):
            drift_metrics = DriftMetrics(**drift_metrics)
        
        # Initialize context if not provided
        drift_context = drift_context or {}
        
        # Determine drift severity
        severity, severity_confidence = self._determine_drift_severity(drift_metrics)
        
        # Determine appropriate response action
        action, action_reasoning = self._determine_response_action(severity, drift_metrics, drift_context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(severity, action, drift_metrics)
        
        # Check if escalation is needed
        escalation_needed = self._check_escalation_needed(severity, drift_context)
        
        # Create response object
        response = DriftResponse(
            severity=severity,
            action=action,
            confidence=severity_confidence,
            reasoning=action_reasoning,
            recommendations=recommendations,
            escalation_needed=escalation_needed,
            metadata={
                'drift_metrics': drift_metrics.__dict__,
                'drift_context': drift_context,
                'evaluation_timestamp': time.time(),
                'config_used': self.config
            }
        )
        
        # Log the response
        self._log_drift_response(response)
        
        # Store in history
        self.response_history.append(response)
        
        return response
    
    def _determine_drift_severity(self, metrics: DriftMetrics) -> Tuple[DriftSeverity, float]:
        """
        Determine drift severity based on multiple metrics
        
        Args:
            metrics: DriftMetrics object
            
        Returns:
            Tuple of (DriftSeverity, confidence_score)
        """
        severity_scores = {}
        
        # 1. Dataset drift score evaluation
        dataset_score = metrics.dataset_drift_score
        if dataset_score >= self.config['major_drift_threshold']:
            severity_scores['dataset'] = (DriftSeverity.MAJOR, 0.9)
        elif dataset_score >= self.config['moderate_drift_threshold']:
            severity_scores['dataset'] = (DriftSeverity.MODERATE, 0.8)
        elif dataset_score >= self.config['minor_drift_threshold']:
            severity_scores['dataset'] = (DriftSeverity.MINOR, 0.7)
        else:
            severity_scores['dataset'] = (DriftSeverity.NONE, 0.9)
        
        # 2. Feature drift count evaluation
        if metrics.total_features > 0:
            feature_drift_pct = metrics.feature_drift_count / metrics.total_features
            if feature_drift_pct >= self.config['major_feature_drift_pct']:
                severity_scores['features'] = (DriftSeverity.MAJOR, 0.8)
            elif feature_drift_pct >= self.config['moderate_feature_drift_pct']:
                severity_scores['features'] = (DriftSeverity.MODERATE, 0.7)
            elif feature_drift_pct >= self.config['minor_feature_drift_pct']:
                severity_scores['features'] = (DriftSeverity.MINOR, 0.6)
            else:
                severity_scores['features'] = (DriftSeverity.NONE, 0.8)
        
        # 3. Concept drift evaluation
        if metrics.concept_drift_score is not None:
            concept_score = metrics.concept_drift_score
            if concept_score >= self.config['concept_drift_threshold']:
                severity_scores['concept'] = (DriftSeverity.CONCEPT, 0.9)
        
        # 4. Performance degradation evaluation
        if metrics.performance_degradation is not None:
            perf_drop = metrics.performance_degradation
            if perf_drop >= self.config['major_performance_drop']:
                severity_scores['performance'] = (DriftSeverity.MAJOR, 0.9)
            elif perf_drop >= self.config['moderate_performance_drop']:
                severity_scores['performance'] = (DriftSeverity.MODERATE, 0.8)
            elif perf_drop >= self.config['minor_performance_drop']:
                severity_scores['performance'] = (DriftSeverity.MINOR, 0.7)
        
        # Determine overall severity (take the most severe)
        severity_order = [DriftSeverity.CONCEPT, DriftSeverity.MAJOR, DriftSeverity.MODERATE, DriftSeverity.MINOR, DriftSeverity.NONE]
        
        overall_severity = DriftSeverity.NONE
        max_confidence = 0.0
        
        for severity in severity_order:
            matching_scores = [conf for sev, conf in severity_scores.values() if sev == severity]
            if matching_scores:
                overall_severity = severity
                max_confidence = max(matching_scores)
                break
        
        logger.info(f"Determined drift severity: {overall_severity.value} (confidence: {max_confidence:.2f})")
        return overall_severity, max_confidence
    
    def _determine_response_action(
        self, 
        severity: DriftSeverity, 
        metrics: DriftMetrics, 
        context: Dict
    ) -> Tuple[ResponseAction, str]:
        """
        Determine appropriate response action based on severity and context
        
        Args:
            severity: Determined drift severity
            metrics: DriftMetrics object
            context: Drift context dictionary
            
        Returns:
            Tuple of (ResponseAction, reasoning_string)
        """
        # Check cooldown period for retraining
        last_retraining = context.get('last_retraining_timestamp', 0)
        hours_since_last_retrain = (time.time() - last_retraining) / 3600
        in_cooldown = hours_since_last_retrain < self.config['retraining_cooldown_hours']
        
        if severity == DriftSeverity.NONE:
            return ResponseAction.CONTINUE_MONITORING, "No significant drift detected, continuing normal monitoring"
        
        elif severity == DriftSeverity.MINOR:
            return ResponseAction.LOG_AND_MONITOR, "Minor drift detected, increasing monitoring frequency and logging"
        
        elif severity == DriftSeverity.MODERATE:
            if in_cooldown:
                return ResponseAction.LOG_AND_MONITOR, f"Moderate drift detected but retraining cooldown active ({hours_since_last_retrain:.1f}h < {self.config['retraining_cooldown_hours']}h)"
            elif self.config['enable_automatic_retraining']:
                return ResponseAction.INCREMENTAL_RETRAIN, "Moderate drift detected, triggering incremental retraining with recent data"
            else:
                return ResponseAction.LOG_AND_MONITOR, "Moderate drift detected but automatic retraining disabled"
        
        elif severity == DriftSeverity.MAJOR:
            if in_cooldown:
                return ResponseAction.LOG_AND_MONITOR, f"Major drift detected but retraining cooldown active ({hours_since_last_retrain:.1f}h < {self.config['retraining_cooldown_hours']}h)"
            elif self.config['enable_automatic_retraining']:
                return ResponseAction.FULL_RETRAIN, "Major drift detected, triggering full model retraining with cumulative data"
            else:
                return ResponseAction.ARCHITECTURE_REVIEW, "Major drift detected but automatic retraining disabled, manual review needed"
        
        elif severity == DriftSeverity.CONCEPT:
            return ResponseAction.ARCHITECTURE_REVIEW, "Concept drift detected, model architecture and feature engineering review required"
        
        else:
            return ResponseAction.CONTINUE_MONITORING, "Unknown severity level, defaulting to monitoring"
    
    def _generate_recommendations(
        self, 
        severity: DriftSeverity, 
        action: ResponseAction, 
        metrics: DriftMetrics
    ) -> List[str]:
        """
        Generate specific recommendations based on drift analysis
        
        Args:
            severity: Drift severity level
            action: Recommended response action
            metrics: DriftMetrics object
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # General recommendations based on severity
        if severity == DriftSeverity.MINOR:
            recommendations.extend([
                "Increase monitoring frequency to detect trend patterns",
                "Review feature distributions for gradual changes",
                "Consider feature engineering improvements"
            ])
        
        elif severity == DriftSeverity.MODERATE:
            recommendations.extend([
                "Analyze specific features showing highest drift",
                "Consider incremental model updates",
                "Review data quality and collection processes",
                "Implement more frequent model validation"
            ])
        
        elif severity == DriftSeverity.MAJOR:
            recommendations.extend([
                "Conduct comprehensive data analysis",
                "Review feature engineering pipeline",
                "Consider model architecture changes",
                "Implement enhanced monitoring and alerting",
                "Review data collection and preprocessing"
            ])
        
        elif severity == DriftSeverity.CONCEPT:
            recommendations.extend([
                "Analyze target variable relationships",
                "Review model assumptions and architecture",
                "Consider ensemble or adaptive modeling approaches",
                "Implement concept drift detection algorithms",
                "Review business logic and domain expertise"
            ])
        
        # Specific recommendations based on metrics
        if metrics.feature_drift_count > 5:
            recommendations.append(f"High feature drift count ({metrics.feature_drift_count} features): Focus on most important features")
        
        if metrics.prediction_drift_score and metrics.prediction_drift_score > 0.2:
            recommendations.append("High prediction drift: Review model confidence and uncertainty estimates")
        
        if metrics.performance_degradation and metrics.performance_degradation > 0.05:
            recommendations.append(f"Performance degradation detected ({metrics.performance_degradation:.1%}): Prioritize model retraining")
        
        return recommendations
    
    def _check_escalation_needed(self, severity: DriftSeverity, context: Dict) -> bool:
        """
        Check if escalation to human intervention is needed
        
        Args:
            severity: Current drift severity
            context: Drift context
            
        Returns:
            Boolean indicating if escalation is needed
        """
        # Check for consecutive major drift events
        recent_major_count = context.get('consecutive_major_drift_count', 0)
        if recent_major_count >= self.config['escalation_threshold']:
            return True
        
        # Always escalate concept drift
        if severity == DriftSeverity.CONCEPT:
            return True
        
        # Check for system health issues
        if context.get('retraining_failures', 0) > 2:
            return True
        
        return False
    
    def _log_drift_response(self, response: DriftResponse):
        """
        Log drift response for audit and monitoring
        
        Args:
            response: DriftResponse object to log
        """
        log_entry = {
            'timestamp': time.time(),
            'severity': response.severity.value,
            'action': response.action.value,
            'confidence': response.confidence,
            'reasoning': response.reasoning,
            'recommendations': response.recommendations,
            'escalation_needed': response.escalation_needed,
            'metadata': response.metadata
        }
        
        # Log to standard logger
        logger.info(f"Drift Response: {response.severity.value} -> {response.action.value} (confidence: {response.confidence:.2f})")
        logger.info(f"Reasoning: {response.reasoning}")
        if response.recommendations:
            logger.info(f"Recommendations: {', '.join(response.recommendations[:3])}")  # Log first 3 recommendations
        
        # Log to S3 if configured and available
        if self.s3_client and self.config.get('drift_response_log_prefix'):
            try:
                timestamp = int(time.time())
                log_key = f"{self.config['drift_response_log_prefix']}response_{timestamp}.json"
                
                self.s3_client.put_object(
                    Bucket=self.config['s3_bucket'],
                    Key=log_key,
                    Body=json.dumps(log_entry, indent=2),
                    ContentType='application/json'
                )
                logger.info(f"Response logged to S3: {log_key}")
            except Exception as e:
                logger.warning(f"Failed to log response to S3: {e}")
    
    def get_response_history(self, limit: int = 10) -> List[DriftResponse]:
        """
        Get recent drift response history
        
        Args:
            limit: Maximum number of responses to return
            
        Returns:
            List of recent DriftResponse objects
        """
        return self.response_history[-limit:]
    
    def update_config(self, new_config: Dict):
        """
        Update configuration parameters
        
        Args:
            new_config: Dictionary with configuration updates
        """
        self.config.update(new_config)
        logger.info(f"Configuration updated: {list(new_config.keys())}")

# Utility functions for Airflow integration
def evaluate_drift_for_airflow(**kwargs) -> Dict:
    """
    Airflow-compatible function to evaluate drift response
    
    Args:
        kwargs: Airflow context containing drift metrics and context
        
    Returns:
        Dictionary with response information for downstream tasks
    """
    # Extract drift metrics from XCom
    ti = kwargs['ti']
    drift_results = ti.xcom_pull(task_ids='run_drift_detection')
    
    if not drift_results:
        logger.warning("No drift results found from upstream task")
        return {
            'action': ResponseAction.CONTINUE_MONITORING.value,
            'severity': DriftSeverity.NONE.value,
            'reasoning': 'No drift detection results available'
        }
    
    # Create drift metrics object
    metrics = DriftMetrics(
        dataset_drift_score=drift_results.get('dataset_drift_score', 0.0),
        feature_drift_count=drift_results.get('feature_drift_count', 0),
        total_features=drift_results.get('total_features', 1),
        concept_drift_score=drift_results.get('concept_drift_score'),
        prediction_drift_score=drift_results.get('prediction_drift_score'),
        performance_degradation=drift_results.get('performance_degradation'),
        confidence_score=drift_results.get('confidence_score')
    )
    
    # Get drift context
    drift_context = kwargs.get('dag_run', {}).conf or {}
    
    # Initialize handler and evaluate response
    handler = DriftResponseHandler()
    response = handler.evaluate_drift_response(metrics, drift_context)
    
    # Return serializable response for XCom
    return {
        'action': response.action.value,
        'severity': response.severity.value,
        'confidence': response.confidence,
        'reasoning': response.reasoning,
        'recommendations': response.recommendations,
        'escalation_needed': response.escalation_needed,
        'response_timestamp': time.time()
    }

if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test drift response handler")
    parser.add_argument('--dataset-drift-score', type=float, default=0.1, help='Dataset drift score')
    parser.add_argument('--feature-drift-count', type=int, default=2, help='Number of features with drift')
    parser.add_argument('--total-features', type=int, default=20, help='Total number of features')
    parser.add_argument('--concept-drift-score', type=float, help='Concept drift score')
    
    args = parser.parse_args()
    
    # Create test metrics
    test_metrics = DriftMetrics(
        dataset_drift_score=args.dataset_drift_score,
        feature_drift_count=args.feature_drift_count,
        total_features=args.total_features,
        concept_drift_score=args.concept_drift_score
    )
    
    # Test the handler
    handler = DriftResponseHandler()
    response = handler.evaluate_drift_response(test_metrics)
    
    print(f"\nDrift Response Evaluation:")
    print(f"Severity: {response.severity.value}")
    print(f"Action: {response.action.value}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Recommendations: {response.recommendations}")
    print(f"Escalation needed: {response.escalation_needed}") 