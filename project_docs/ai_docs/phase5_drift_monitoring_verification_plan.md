# Phase 5 Drift Monitoring - Comprehensive Verification Plan

## Overview

This plan verifies that all Phase 5 drift monitoring components work together properly, from basic functionality through complex end-to-end workflows. We'll test each component individually and then verify integration points.

## Components to Verify

### Core Scripts (Week 1-3)
- `scripts/monitor_drift.py` - Core drift detection with Evidently AI + advanced statistical methods
- `scripts/drift_injection.py` - Synthetic drift injection capabilities
- `scripts/drift_dashboard.py` - Streamlit visualization dashboard

### Response System (Week 4)
- `scripts/drift_response_handler.py` - Graduated response system
- `mlops-services/dags/training_pipeline_dag.py` - Enhanced with drift-aware retraining

### Advanced Integration (Week 5)
- `mlops-services/dags/drift_monitoring_dag.py` - Enhanced monitoring DAG v2
- `scripts/batch_processing_simulation.py` - Healthcare-specific batch simulation
- `scripts/drift_monitoring_error_handler.py` - Production error handling
- `scripts/test_end_to_end_drift_pipeline.py` - Comprehensive testing framework

## Verification Phases

### Phase 1: Environment & Infrastructure Verification

#### Step 1.1: Environment Setup Verification
**Objective**: Confirm all environment variables and infrastructure are properly configured

**Commands**:
```bash
# Verify environment variables
docker exec airflow-webserver env | grep DRIFT | sort

# Check S3 structure
aws s3 ls s3://health-predict-mlops-f9ac6509/drift_monitoring/ --recursive

# Verify MLflow experiment exists
docker exec airflow-scheduler python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
experiments = mlflow.search_experiments()
for exp in experiments:
    if 'Drift' in exp.name:
        print(f'Found: {exp.name} (ID: {exp.experiment_id})')
"
```

**Expected Results**:
- [ ] All DRIFT_* environment variables present
- [ ] S3 directories exist: `batch_data/`, `reports/`, `reference_data/`, `retraining_data/`
- [ ] MLflow experiment `HealthPredict_Drift_Monitoring` exists

#### Step 1.2: Reference Data Preparation
**Objective**: Ensure reference data is available for drift detection

**Commands**:
```bash
# Check if reference data exists
aws s3 ls s3://health-predict-mlops-f9ac6509/drift_monitoring/reference_data/

# If missing, create reference data
python scripts/split_data.py
aws s3 cp data/initial_train.csv s3://health-predict-mlops-f9ac6509/drift_monitoring/reference_data/
```

**Expected Results**:
- [ ] Reference data file exists in S3
- [ ] File size ~2.5MB with 14,247 rows

### Phase 2: Individual Component Testing

#### Step 2.1: Core Drift Detection Script Testing
**Objective**: Verify `monitor_drift.py` works with basic and advanced methods

**Test 1: Basic Drift Detection**
```bash
# Create test batch from future data
head -1001 data/future_data.csv > /tmp/test_batch.csv
aws s3 cp /tmp/test_batch.csv s3://health-predict-mlops-f9ac6509/drift_monitoring/batch_data/

# Run basic drift detection
docker exec airflow-scheduler python /opt/airflow/scripts/monitor_drift.py \
  --s3_new_data_path "s3://health-predict-mlops-f9ac6509/drift_monitoring/batch_data/test_batch.csv" \
  --s3_reference_data_path "s3://health-predict-mlops-f9ac6509/drift_monitoring/reference_data/initial_train.csv" \
  --s3_evidently_reports_path "s3://health-predict-mlops-f9ac6509/drift_monitoring/reports/test_basic" \
  --target_column "readmitted_binary"
```

**Expected Results**:
- [ ] Script executes without errors
- [ ] Returns `DRIFT_DETECTED` or `NO_DRIFT`
- [ ] Creates MLflow run in drift monitoring experiment
- [ ] Uploads HTML report to S3

**Test 2: Advanced Statistical Methods**
```bash
# Run with advanced methods enabled
docker exec airflow-scheduler python /opt/airflow/scripts/monitor_drift.py \
  --s3_new_data_path "s3://health-predict-mlops-f9ac6509/drift_monitoring/batch_data/test_batch.csv" \
  --s3_reference_data_path "s3://health-predict-mlops-f9ac6509/drift_monitoring/reference_data/initial_train.csv" \
  --s3_evidently_reports_path "s3://health-predict-mlops-f9ac6509/drift_monitoring/reports/test_advanced" \
  --target_column "readmitted_binary" \
  --enable_advanced_methods
```

**Expected Results**:
- [ ] Additional statistical metrics logged (KS-test, PSI, Wasserstein, JS divergence)
- [ ] Ensemble confidence score calculated
- [ ] Feature-level drift metrics available

#### Step 2.2: Synthetic Drift Injection Testing
**Objective**: Verify synthetic drift capabilities work

**Commands**:
```bash
# Test drift injection capabilities
docker exec airflow-scheduler python -c "
import sys
sys.path.append('/opt/airflow/scripts')
from drift_injection import inject_covariate_drift, inject_concept_drift, create_drift_scenario
import pandas as pd

# Load reference data
ref_data = pd.read_csv('/opt/airflow/data/initial_train.csv')
print(f'Reference data shape: {ref_data.shape}')

# Test covariate drift injection
drifted_data = inject_covariate_drift(ref_data, drift_type='shift', intensity=0.3)
print(f'Covariate drift applied: {drifted_data.shape}')

# Test concept drift injection
concept_drifted = inject_concept_drift(ref_data, 'readmitted_binary', drift_type='gradual', intensity=0.2)
print(f'Concept drift applied: {concept_drifted.shape}')

print('✅ Drift injection functions working')
"
```

**Expected Results**:
- [ ] Drift injection functions execute without errors
- [ ] Data shapes preserved after drift injection
- [ ] Different drift types (shift, scale, noise, concept) work

#### Step 2.3: Dashboard Testing
**Objective**: Verify Streamlit dashboard loads and displays drift metrics

**Commands**:
```bash
# Test dashboard dependencies
docker exec airflow-scheduler python -c "
import streamlit as st
import plotly.graph_objects as go
import mlflow
print('✅ Dashboard dependencies available')
"

# Run dashboard script check (syntax verification)
docker exec airflow-scheduler python -m py_compile /opt/airflow/scripts/drift_dashboard.py
```

**Expected Results**:
- [ ] All dashboard dependencies available
- [ ] Script compiles without syntax errors
- [ ] Can import MLflow experiment data

### Phase 3: DAG Integration Testing

#### Step 3.1: DAG Parsing and Structure Verification
**Objective**: Verify DAGs parse correctly and have proper structure

**Commands**:
```bash
# Check DAG parsing
docker exec airflow-scheduler airflow dags list | grep -E "(drift|training)"

# Test DAG structure
docker exec airflow-scheduler airflow tasks list drift_monitoring_dag
docker exec airflow-scheduler airflow tasks list health_predict_training_hpo
```

**Expected Results**:
- [ ] `drift_monitoring_dag` parses successfully
- [ ] `health_predict_training_hpo` parses successfully
- [ ] All expected tasks present in DAGs

#### Step 3.2: Basic DAG Execution Testing
**Objective**: Test individual DAG task execution

**Test 1: Drift Monitoring DAG Tasks**
```bash
# Test batch simulation task
docker exec airflow-scheduler airflow tasks test drift_monitoring_dag simulate_data_batch 2025-01-24

# Test drift detection task (requires batch data from previous task)
# Note: This may fail initially due to XCom dependencies - that's expected
```

**Test 2: Training Pipeline Integration**
```bash
# Verify training DAG can handle drift context
docker exec airflow-scheduler airflow dags show health_predict_training_hpo
```

**Expected Results**:
- [ ] Individual tasks can be tested
- [ ] No syntax or import errors
- [ ] Proper error handling when dependencies missing

### Phase 4: Error Handling and Recovery Testing

#### Step 4.1: Error Handler Component Testing
**Objective**: Verify error classification and recovery mechanisms

**Commands**:
```bash
# Test error handler functionality
docker exec airflow-scheduler python -c "
import sys
sys.path.append('/opt/airflow/scripts')
from drift_monitoring_error_handler import DriftMonitoringErrorHandler

handler = DriftMonitoringErrorHandler()

# Test error classification
test_errors = [
    'S3 connection timeout',
    'MLflow server unavailable',
    'Memory allocation failed',
    'Invalid data format'
]

for error in test_errors:
    try:
        severity, category, action = handler.classify_error(Exception(error))
        print(f'Error: {error[:30]}... -> {severity}/{category}/{action}')
    except Exception as e:
        print(f'Error in classification: {e}')

print('✅ Error handler classification working')
"
```

**Expected Results**:
- [ ] Error classification works for different error types
- [ ] Appropriate severity levels assigned
- [ ] Recovery actions suggested

#### Step 4.2: Circuit Breaker Testing
**Objective**: Test circuit breaker protection for external services

**Commands**:
```bash
# Test circuit breaker functionality
docker exec airflow-scheduler python -c "
import sys
sys.path.append('/opt/airflow/scripts')
from drift_monitoring_error_handler import DriftMonitoringErrorHandler

handler = DriftMonitoringErrorHandler()

# Test circuit breaker states
print(f'MLflow circuit breaker: {handler.mlflow_circuit_breaker.is_closed}')
print(f'S3 circuit breaker: {handler.s3_circuit_breaker.is_closed}')

print('✅ Circuit breakers initialized')
"
```

**Expected Results**:
- [ ] Circuit breakers initialize properly
- [ ] Default state is closed (allowing traffic)
- [ ] Can check circuit breaker status

### Phase 5: End-to-End Workflow Testing

#### Step 5.1: Simple End-to-End Flow
**Objective**: Test complete drift detection → response workflow

**Commands**:
```bash
# Run simplified end-to-end test
docker exec airflow-scheduler python /opt/airflow/scripts/test_end_to_end_drift_pipeline.py \
  --scenario no_drift_baseline \
  --batch_size 500 \
  --timeout 300
```

**Expected Results**:
- [ ] Test executes without errors
- [ ] Creates test data batch
- [ ] Runs drift detection
- [ ] Evaluates response appropriately
- [ ] Logs results to MLflow

#### Step 5.2: Drift Response Integration
**Objective**: Test drift detection triggers appropriate responses

**Test 1: Minor Drift Response**
```bash
# Test minor drift scenario
docker exec airflow-scheduler python /opt/airflow/scripts/test_end_to_end_drift_pipeline.py \
  --scenario minor_covariate_drift \
  --batch_size 1000 \
  --timeout 600
```

**Test 2: Major Drift with Retraining Trigger**
```bash
# Test major drift scenario (may trigger retraining)
docker exec airflow-scheduler python /opt/airflow/scripts/test_end_to_end_drift_pipeline.py \
  --scenario moderate_concept_drift \
  --batch_size 1000 \
  --timeout 900
```

**Expected Results**:
- [ ] Different drift scenarios produce appropriate responses
- [ ] Minor drift results in monitoring continuation
- [ ] Major drift may trigger retraining workflow
- [ ] All scenarios log properly to MLflow

#### Step 5.3: Batch Processing Simulation
**Objective**: Test healthcare-specific batch arrival patterns

**Commands**:
```bash
# Test batch processing simulation
docker exec airflow-scheduler python /opt/airflow/scripts/batch_processing_simulation.py \
  --pattern healthcare_steady \
  --duration 30 \
  --batch_count 5 \
  --dry_run
```

**Expected Results**:
- [ ] Different arrival patterns work (steady, surge, emergency, maintenance)
- [ ] Realistic timing variations applied
- [ ] Data quality variations simulated
- [ ] Batch metadata tracked properly

### Phase 6: Performance and Load Testing

#### Step 6.1: Parallel Processing Testing
**Objective**: Verify parallel batch processing capabilities

**Commands**:
```bash
# Test parallel processing capability
docker exec airflow-scheduler python -c "
import sys
sys.path.append('/opt/airflow/scripts')
from test_end_to_end_drift_pipeline import EndToEndDriftTester

tester = EndToEndDriftTester()

# Test concurrent batch processing
results = tester.run_load_test(
    concurrent_batches=3,
    duration_minutes=5,
    batch_size=500
)

print(f'Load test results: {results}')
"
```

**Expected Results**:
- [ ] Multiple batches can be processed simultaneously
- [ ] System remains stable under load
- [ ] Performance metrics collected
- [ ] No resource conflicts

### Phase 7: Dashboard and Visualization Testing

#### Step 7.1: Dashboard Data Loading
**Objective**: Verify dashboard can load and display drift monitoring data

**Commands**:
```bash
# Test dashboard data loading capabilities
docker exec airflow-scheduler python -c "
import sys
sys.path.append('/opt/airflow/scripts')
import mlflow
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri('http://mlflow:5000')

# Try to load drift monitoring experiment data
try:
    experiment = mlflow.get_experiment_by_name('HealthPredict_Drift_Monitoring')
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
        print(f'Found {len(runs)} drift monitoring runs')
        if len(runs) > 0:
            print('Recent run metrics:', runs.columns.tolist())
            print('✅ Dashboard data loading possible')
        else:
            print('⚠️  No drift monitoring runs found - run some drift detection first')
    else:
        print('❌ Drift monitoring experiment not found')
except Exception as e:
    print(f'❌ Error loading experiment data: {e}')
"
```

**Expected Results**:
- [ ] Can connect to MLflow experiment
- [ ] Drift monitoring runs are available
- [ ] Run metrics can be loaded for visualization

### Phase 8: Integration Validation

#### Step 8.1: Complete System Health Check
**Objective**: Verify all system components are healthy and integrated

**Commands**:
```bash
# Run comprehensive system health check
docker exec airflow-scheduler python -c "
import sys
sys.path.append('/opt/airflow/scripts')
from drift_monitoring_error_handler import DriftMonitoringErrorHandler

handler = DriftMonitoringErrorHandler()
health_status = handler.check_system_health()

print('=== System Health Check ===')
for component, status in health_status.items():
    print(f'{component}: {status}')

print('✅ System health check completed')
"
```

**Expected Results**:
- [ ] All components report healthy status
- [ ] MLflow, S3, Airflow connectivity confirmed
- [ ] No critical errors reported

#### Step 8.2: Full Workflow Demonstration
**Objective**: Execute complete drift monitoring workflow end-to-end

**Commands**:
```bash
# Full workflow demonstration
echo "=== Phase 5 Drift Monitoring - Full Workflow Demonstration ==="

# 1. Trigger drift monitoring DAG
docker exec airflow-scheduler airflow dags trigger drift_monitoring_dag

# 2. Monitor DAG execution
echo "Monitor execution in Airflow UI: http://localhost:8080"
echo "DAG: drift_monitoring_dag"

# 3. Check MLflow for results
echo "Monitor results in MLflow UI: http://localhost:5000"
echo "Experiment: HealthPredict_Drift_Monitoring"

# 4. Check S3 for reports
echo "Reports will be stored in: s3://health-predict-mlops-f9ac6509/drift_monitoring/reports/"
```

**Expected Results**:
- [ ] DAG triggers successfully
- [ ] All DAG tasks execute without errors
- [ ] Drift detection results logged to MLflow
- [ ] Reports uploaded to S3
- [ ] Appropriate response actions taken based on drift level

## Success Criteria

### Phase 1 (Infrastructure): ✅
- [ ] All environment variables configured
- [ ] S3 structure exists and accessible
- [ ] MLflow experiment created and accessible
- [ ] Reference data available

### Phase 2 (Components): ✅
- [ ] Core drift detection script works
- [ ] Advanced statistical methods functional
- [ ] Synthetic drift injection works
- [ ] Dashboard dependencies available

### Phase 3 (DAGs): ✅
- [ ] DAGs parse without errors
- [ ] Individual tasks executable
- [ ] Proper task dependencies

### Phase 4 (Error Handling): ✅
- [ ] Error classification working
- [ ] Circuit breakers functional
- [ ] Recovery mechanisms available

### Phase 5 (End-to-End): ✅
- [ ] Simple workflow executes
- [ ] Different drift scenarios handled appropriately
- [ ] Batch simulation works

### Phase 6 (Performance): ✅
- [ ] Parallel processing functional
- [ ] System stable under load
- [ ] Performance metrics collected

### Phase 7 (Visualization): ✅
- [ ] Dashboard can load data
- [ ] MLflow experiment accessible

### Phase 8 (Integration): ✅
- [ ] System health check passes
- [ ] Full workflow demonstration successful

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Environment Variables Missing
**Problem**: `KeyError` for DRIFT_* variables
**Solution**: 
```bash
# Check docker-compose.yml for missing environment variables
# Restart containers after adding variables
docker-compose down && docker-compose up -d
```

#### 2. S3 Access Issues
**Problem**: `NoCredentialsError` or `AccessDenied`
**Solution**: 
```bash
# Verify AWS credentials in container
docker exec airflow-scheduler aws sts get-caller-identity
```

#### 3. MLflow Connection Issues
**Problem**: `ConnectionError` to MLflow server
**Solution**: 
```bash
# Check MLflow service status
docker-compose ps mlflow
# Restart MLflow if needed
docker-compose restart mlflow
```

#### 4. DAG Import Errors
**Problem**: DAGs not parsing due to import errors
**Solution**: 
```bash
# Check Python path and dependencies
docker exec airflow-scheduler python -c "import sys; print(sys.path)"
# Install missing dependencies if needed
```

#### 5. Memory Issues During Testing
**Problem**: `MemoryError` during large batch processing
**Solution**: 
- Reduce batch sizes in tests
- Increase Docker memory allocation
- Use streaming processing for large datasets

## Next Steps After Verification

1. **Document Issues**: Record any failures and their resolutions
2. **Performance Optimization**: Address any performance bottlenecks found
3. **Integration Refinement**: Fix any integration issues discovered
4. **Production Readiness**: Prepare for Phase 5C production polish
5. **Demonstration Material**: Create demo scripts based on working workflows 