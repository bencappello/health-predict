# Unified DAG Implementation Plan: Continuous Model Improvement Pipeline

## Overview

This document outlines the comprehensive implementation plan for transitioning from the current split DAG architecture (separate training and deployment DAGs) to a unified **Continuous Model Improvement Pipeline**. This approach aligns with industry MLOps best practices by treating model improvement as an atomic operation that automatically deploys better models to production.

## 1. Core Concept & Philosophy

### Current State Problems
- **Trained but Never Deployed**: Better models can sit unused in MLflow registry
- **Manual Intervention Required**: Deployment requires separate DAG trigger
- **Fragmented Workflow**: Model quality and production deployment are disconnected
- **Risk of Stale Production**: No automated promotion of superior models

### Unified Approach Benefits
- **Atomic Operations**: Model improvement automatically results in production deployment
- **Quality Gates**: Sophisticated decision logic determines deployment worthiness
- **Continuous Improvement**: Production always runs the best available model
- **Zero Manual Intervention**: Fully automated model-to-production pipeline
- **Risk Mitigation**: Comprehensive testing before deployment, rollback capabilities

## 2. DAG Architecture Design

### 2.1 DAG Name & Purpose
- **DAG ID**: `health_predict_continuous_improvement`
- **Purpose**: End-to-end pipeline from training to production deployment
- **Trigger**: Manual, scheduled, or drift-detection initiated

### 2.2 High-Level Task Flow
```
Data Preparation → Model Training & HPO → Quality Gates → [Decision Branch]
                                                              ↓
                                            [Deploy Path] → API Testing → Deployment → Verification → Notification
                                                              ↓
                                            [Skip Path] → Log Decision → Notification
```

### 2.3 Task Dependencies Structure
```
start → prepare_training_data
     → run_training_and_hpo
     → evaluate_model_performance
     → compare_against_production
     → [Branching Operator with XCom decision]
         ├── deploy_branch:
         │   → register_and_promote_model
         │   → build_api_image
         │   → test_api_locally
         │   → push_to_ecr
         │   → deploy_to_kubernetes
         │   → verify_deployment
         │   → post_deployment_health_check
         │   → notify_deployment_success
         └── skip_branch:
             → log_skip_decision
             → notify_no_deployment
     → end
```

## 3. Detailed Task Specifications

### 3.1 Core Training Tasks

#### Task: `prepare_training_data`
- **Function**: Load and prepare training data from S3
- **Inputs**: S3 bucket configuration, data paths
- **Outputs**: XCom with data statistics, preparation metadata
- **Error Handling**: Retry on S3 connection issues, fail DAG on data corruption

#### Task: `run_training_and_hpo`
- **Function**: Execute hyperparameter optimization for multiple model types
- **Behavior**: Same as current training DAG implementation
- **Outputs**: XCom with best model metrics for each model type
- **MLflow Integration**: Log all experiments and mark best models

#### Task: `evaluate_model_performance`
- **Function**: Consolidate and evaluate all trained models
- **Logic**: 
  - Compare performance across model types (RF, XGBoost, LR)
  - Select single best performer based on primary metric (F1 score)
  - Calculate confidence intervals and statistical significance
- **Outputs**: XCom with champion model details and performance metrics

### 3.2 Quality Gates & Decision Logic

#### Task: `compare_against_production`
- **Function**: Implement sophisticated decision logic for deployment
- **Decision Criteria**:
  - **Performance Threshold**: New model F1 score > current production + improvement_threshold
  - **Statistical Significance**: Confidence interval analysis
  - **Business Rules**: Minimum sample size, maximum performance degradation tolerance
  - **Model Staleness**: Time since last production model update
- **Configuration Parameters**:
  ```
  improvement_threshold: 0.02  # 2% F1 improvement required
  confidence_level: 0.95
  min_sample_size: 1000
  max_days_since_update: 30  # Force update after 30 days regardless
  ```
- **Outputs**: XCom decision flag and detailed reasoning
- **Business Logic**:
  ```
  if new_f1 > production_f1 + threshold AND statistical_significance:
      return "DEPLOY"
  elif days_since_update > max_days AND new_f1 >= production_f1:
      return "DEPLOY_REFRESH"  # Refresh deployment even without major improvement
  else:
      return "SKIP"
  ```

#### Task: `deployment_decision_branch`
- **Function**: Airflow BranchPythonOperator based on quality gates
- **Logic**: Route to either deployment path or skip path
- **Decision Source**: XCom from `compare_against_production`

### 3.3 Deployment Path Tasks

#### Task: `register_and_promote_model`
- **Function**: Register new champion model in MLflow registry
- **Actions**:
  - Archive current Production model to Archived stage
  - Register new model version
  - Promote to Production stage
  - Update model metadata with deployment timestamp
- **Validation**: Ensure model artifacts and preprocessor are available

#### Task: `build_api_image`
- **Function**: Build Docker image with new model
- **Process**:
  - Generate unique image tag based on model version and timestamp
  - Build Docker image using existing Dockerfile
  - Include model version in environment variables
- **Outputs**: XCom with image tag and build metadata

#### Task: `test_api_locally`
- **Function**: Comprehensive API testing using Docker container approach
- **Implementation**: Enhanced version of current pre-deployment testing
- **Test Suite**:
  - Health endpoint validation
  - Prediction endpoint with various inputs
  - Error handling verification
  - Performance baseline testing
  - Model version verification
- **Pass Criteria**: All tests must pass for deployment to continue

#### Task: `push_to_ecr`
- **Function**: Push validated Docker image to ECR
- **Conditions**: Only execute if API tests pass
- **Error Handling**: Retry on network issues, fail on authentication problems

#### Task: `deploy_to_kubernetes`
- **Function**: Update Kubernetes deployment with new image
- **Process**:
  - Update deployment manifest with new image tag
  - Apply changes to Kubernetes cluster
  - Monitor deployment rollout progress
- **Configuration**: Use existing K8s deployment strategies

#### Task: `verify_deployment`
- **Function**: Verify successful deployment using Kubernetes health checks
- **Implementation**: Enhanced version of current deployment verification
- **Checks**:
  - Pod readiness verification
  - Service endpoint availability
  - Basic health endpoint response
  - Model version confirmation

#### Task: `post_deployment_health_check`
- **Function**: Extended health verification post-deployment
- **Checks**:
  - Multiple health endpoint calls
  - Basic prediction endpoint validation
  - Response time verification
  - Error rate monitoring (5-minute window)

### 3.4 Skip Path Tasks

#### Task: `log_skip_decision`
- **Function**: Log detailed reasoning for not deploying
- **Information Logged**:
  - Performance comparison details
  - Decision criteria evaluation
  - Recommendations for improvement
  - Next suggested training schedule

### 3.5 Notification Tasks

#### Task: `notify_deployment_success` / `notify_no_deployment`
- **Function**: Send notifications about DAG outcome
- **Channels**: Airflow logging, email (if configured), Slack (future)
- **Information**: Model performance, deployment status, next steps

## 4. Environment & System Setup Changes

### 4.1 MLflow Registry Configuration
- **Model Naming Convention**: Ensure consistent naming across model types
- **Stage Management**: Automated promotion/demotion logic
- **Metadata Enhancement**: Add deployment timestamps, performance deltas

### 4.2 Airflow Configuration Updates
- **DAG Configuration**:
  ```
  schedule_interval: None  # Manual/external triggers
  catchup: False
  max_active_runs: 1  # Prevent concurrent improvement pipelines
  default_retries: 2
  retry_delay: timedelta(minutes=5)
  ```
- **Resource Allocation**: Ensure sufficient worker capacity for combined pipeline
- **Timeout Settings**: Extended timeouts for comprehensive testing

### 4.3 Docker Network Configuration
- **Network Isolation**: Ensure test containers can communicate with MLflow service
- **Port Management**: Avoid conflicts during API testing
- **Cleanup Procedures**: Enhanced container cleanup after testing

### 4.4 Kubernetes Integration
- **Deployment Strategy**: Configure rolling updates for zero-downtime deployments
- **Resource Limits**: Set appropriate CPU/memory limits for new deployments
- **Health Probes**: Enhance readiness and liveness probes

## 5. Migration Strategy

### 5.1 Phase 1: Development & Testing
1. **Create New DAG File**: `health_predict_continuous_improvement.py`
2. **Implement Core Tasks**: Start with training and decision logic
3. **Test Decision Logic**: Use mock data to verify quality gates
4. **Validate MLflow Integration**: Ensure proper model registration flow

### 5.2 Phase 2: Integration Testing
1. **End-to-End Testing**: Run complete pipeline in development environment
2. **API Testing Enhancement**: Integrate and improve existing test suite
3. **Kubernetes Deployment Testing**: Verify deployment procedures
4. **Rollback Procedures**: Implement and test failure recovery

### 5.3 Phase 3: Production Migration
1. **Pause Current DAGs**: Temporarily disable split DAGs
2. **Deploy Unified DAG**: Make new DAG available in production
3. **Initial Manual Run**: Execute with manual trigger and monitoring
4. **Validation**: Verify all components work correctly
5. **Cleanup**: Remove old DAG files after successful validation

### 5.4 Phase 4: Enhancement
1. **Performance Optimization**: Tune timeouts and resource allocation
2. **Monitoring Integration**: Add comprehensive logging and metrics
3. **Advanced Features**: Implement A/B testing capabilities (future)

## 6. Configuration Management

### 6.1 Environment Variables
```
# Quality Gate Configuration
MODEL_IMPROVEMENT_THRESHOLD=0.02
CONFIDENCE_LEVEL=0.95
MIN_SAMPLE_SIZE=1000
MAX_DAYS_SINCE_UPDATE=30

# Deployment Configuration
DOCKER_IMAGE_REGISTRY=your-ecr-registry
K8S_NAMESPACE=default
K8S_DEPLOYMENT_NAME=health-predict-api

# Testing Configuration
API_TEST_TIMEOUT=300
API_TEST_RETRIES=3
HEALTH_CHECK_INTERVAL=30
```

### 6.2 DAG Parameters
```python
# Configurable parameters for different environments
dag_params = {
    'improvement_threshold': 0.02,
    'force_deployment': False,  # Override quality gates
    'skip_api_tests': False,    # For emergency deployments
    'notification_channels': ['airflow', 'email'],
    'rollback_on_failure': True
}
```

## 7. Error Handling & Recovery

### 7.1 Failure Scenarios & Responses

#### Training Failure
- **Response**: Retry with exponential backoff
- **Escalation**: Alert on repeated failures
- **Recovery**: Maintain current production model

#### Quality Gate Failure
- **Response**: Log detailed reasoning and continue with skip path
- **Recovery**: No deployment, current production model remains

#### API Testing Failure
- **Response**: Halt deployment, maintain current production
- **Escalation**: Immediate alert for manual investigation
- **Recovery**: Rollback to previous known-good state

#### Deployment Failure
- **Response**: Automatic rollback to previous deployment
- **Verification**: Confirm rollback success
- **Escalation**: Alert for manual intervention

### 7.2 Rollback Procedures
- **Automated Rollback**: Triggered by health check failures
- **Manual Rollback**: Available via Airflow UI
- **Rollback Verification**: Comprehensive health checks post-rollback
- **State Management**: Update MLflow registry to reflect rollback

## 8. Monitoring & Observability

### 8.1 Key Metrics to Track
- **Pipeline Performance**: Execution time, success rate, failure points
- **Model Performance**: F1 score improvements, deployment frequency
- **Deployment Health**: Success rate, rollback frequency
- **Business Impact**: Prediction accuracy in production

### 8.2 Logging Strategy
- **Structured Logging**: JSON format for easy parsing
- **Performance Logging**: Detailed timing for each task
- **Decision Logging**: Complete reasoning for deployment decisions
- **Error Logging**: Comprehensive error context and stack traces

### 8.3 Alerting Configuration
- **Success Notifications**: Model deployments, performance improvements
- **Warning Alerts**: Quality gate near-misses, performance degradation
- **Critical Alerts**: Pipeline failures, deployment failures, health check failures

## 9. Testing Strategy

### 9.1 Unit Testing
- **Quality Gate Logic**: Test decision algorithms with various scenarios
- **MLflow Integration**: Mock MLflow calls and test registry operations
- **Kubernetes Operations**: Mock K8s client calls

### 9.2 Integration Testing
- **End-to-End Pipeline**: Complete DAG execution in test environment
- **API Testing**: Enhanced test suite with edge cases
- **Deployment Testing**: Kubernetes deployment verification

### 9.3 Performance Testing
- **Training Performance**: Baseline execution times
- **API Performance**: Response time and throughput validation
- **Deployment Performance**: Rollout time and resource usage

## 10. Future Enhancements

### 10.1 Advanced Features
- **A/B Testing**: Gradual model rollout capabilities
- **Multi-Environment**: Support for staging and production environments
- **Advanced Quality Gates**: Business metric integration
- **Automated Drift Detection**: Integration with monitoring pipeline

### 10.2 Scalability Improvements
- **Parallel Training**: Multi-model training optimization
- **Resource Optimization**: Dynamic resource allocation
- **Caching**: Model artifact and preprocessing caching

### 10.3 Operational Enhancements
- **Dashboard Integration**: Real-time pipeline monitoring
- **Slack/Teams Integration**: Advanced notification systems
- **Audit Trail**: Comprehensive deployment history tracking

## 11. Success Criteria

### 11.1 Technical Success Metrics
- **Pipeline Reliability**: >95% success rate
- **Deployment Speed**: <30 minutes end-to-end
- **Zero-Downtime**: No service interruptions during deployments
- **Rollback Capability**: <5 minutes recovery time

### 11.2 Business Success Metrics
- **Model Freshness**: Production model updated within defined thresholds
- **Performance Improvement**: Measurable prediction accuracy gains
- **Operational Efficiency**: Reduced manual intervention requirements
- **Risk Mitigation**: No degraded production models

## 12. Implementation Timeline

### Week 1: Foundation
- Design and document unified DAG architecture
- Implement core training and evaluation tasks
- Develop quality gate decision logic

### Week 2: Integration
- Implement deployment path tasks
- Integrate API testing and Kubernetes deployment
- Develop error handling and rollback procedures

### Week 3: Testing & Validation
- Comprehensive testing of all components
- Performance optimization and tuning
- Documentation and runbook creation

### Week 4: Migration & Go-Live
- Production migration from split DAGs
- Initial production runs with monitoring
- Performance validation and optimization

## 13. Debugging Optimizations (Temporary)

During development and debugging phases, the following temporary optimizations are applied to speed up iteration cycles:

### 13.1 Training Speed Optimizations
**Purpose**: Enable rapid DAG debugging without waiting for full model training

**Changes Applied**:
- **Model Selection**: Train only Logistic Regression (skip RandomForest and XGBoost)
- **HPO Trials**: Reduced from 2 to 1 trial (`RAY_NUM_SAMPLES: '1'`)
- **Training Iterations**: Reduced from 10 to 2 epochs (`RAY_MAX_EPOCHS: '2'`)
- **Early Stopping**: Reduced patience from 5 to 1 (`RAY_GRACE_PERIOD: '1'`)

**Implementation**: Modify `scripts/train_model.py` to focus on LogisticRegression model configuration only.

### 13.2 Decision Logic Override
**Purpose**: Force deployment path execution for testing downstream tasks

**Changes Applied**:
- **Quality Gate Bypass**: `compare_against_production` always returns "DEPLOY"
- **Reasoning**: Skip actual performance comparison to ensure deployment path testing
- **Implementation**: Temporary override in decision logic to return hardcoded "DEPLOY" decision

### 13.3 Reversion Strategy
**Critical**: These optimizations MUST be reverted before production use:

1. **Training Configuration**:
   - Restore full model suite: LogisticRegression, RandomForest, XGBoost
   - Reset HPO parameters: `RAY_NUM_SAMPLES='2'`, `RAY_MAX_EPOCHS='10'`, `RAY_GRACE_PERIOD='5'`

2. **Decision Logic**:
   - Restore actual performance comparison logic
   - Re-enable quality gates and statistical significance testing
   - Remove hardcoded "DEPLOY" return

3. **Validation**:
   - Full end-to-end testing with production settings
   - Verify quality gates work correctly
   - Confirm model performance evaluation accuracy

**Timeline**: Revert after successful DAG flow validation (estimated 1-2 days)

This comprehensive plan provides the roadmap for implementing a production-ready unified DAG that embodies MLOps best practices and ensures continuous model improvement with minimal operational overhead. 