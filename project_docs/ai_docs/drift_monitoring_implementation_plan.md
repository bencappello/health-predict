# Health Predict Drift Monitoring Implementation Plan

## 📋 **Executive Summary**

This document outlines the implementation plan for Phase 5 of the Health Predict MLOps project: **Data Drift Detection and Automated Retraining**. Based on comprehensive assessment of the current system and planned architecture, this plan includes critical improvements and a detailed execution roadmap.

**Current Status**: End of Phase 4 - Fully operational XGBoost MLOps pipeline with CI/CD automation  
**Next Phase**: Phase 5 - Drift monitoring with automated retraining loop  
**Timeline**: 4-6 weeks for complete implementation  
**Last Updated**: 2025-01-24 - Updated based on recent progress assessment

---

## 🎯 **Phase 5 Overview**

### **Objective**
Implement comprehensive data drift detection and automated model retraining to ensure sustained model performance in production by:
- Detecting statistical and concept drift in incoming data
- Triggering automated retraining when significant drift is detected
- Maintaining model performance through continuous monitoring
- Providing visualization and alerting for drift events

### **Core Components**
1. **Drift Detection Script**: `scripts/monitor_drift.py` using Evidently AI
2. **Monitoring DAG**: `monitoring_retraining_dag.py` for orchestration
3. **Data Simulation**: Enhanced future data batching with synthetic drift
4. **Integration Layer**: Connection to existing training and deployment pipelines
5. **Monitoring Dashboard**: MLflow-based drift visualization and alerting

---

## ⚠️ **Critical Issues Identified & Solutions**

### **Issue 1: Missing Core Dependencies**
**Problem**: Evidently AI not installed, core scripts don't exist  
**Solution**: Install dependencies and create foundational scripts

### **Issue 2: Data Simulation Realism**
**Problem**: Current data split assumes temporal order but dataset may not be chronologically ordered  
**Solution**: Implement synthetic drift injection for controlled testing scenarios

### **Issue 3: Integration Gaps**
**Problem**: No clear handoff between drift detection and existing training pipeline  
**Solution**: Modify existing DAGs to handle drift-triggered retraining with data combination logic

### **Issue 4: Missing Production Features**
**Problem**: No dashboards, alerting, or graduated response mechanisms  
**Solution**: Implement multi-level drift responses and monitoring infrastructure

---

## 🚀 **Enhanced Architecture & Strategy**

### **1. Model Retraining Strategy**
When drift is detected, implement **cumulative retraining approach**:

```python
# Retraining Data Combination Strategy
retraining_data = {
    'reference_data': 'initial_20_percent_data',  # Original training data (split_data.py output)
    'processed_batches': 'all_batches_up_to_drift_point',  # Cumulative future data processed
    'strategy': 'cumulative',  # Combine all available data for retraining
    'temporal_validation': True  # Use time-based train/test split within combined data
}

# Example retraining sequence:
# Batch 1 (no drift): Monitor only
# Batch 2 (no drift): Monitor only  
# Batch 3 (drift detected): Retrain on [initial_data + batch_1 + batch_2 + batch_3]
# Batch 4 (no drift): Monitor with new model
# Batch 5 (drift detected): Retrain on [initial_data + batch_1 + batch_2 + batch_3 + batch_4 + batch_5]
```

**Rationale**: Healthcare data benefits from larger training sets. Cumulative approach:
- Preserves historical patterns and relationships
- Increases sample size for better model generalization
- Maintains continuity with original model training approach
- Enables learning from gradual distribution changes over time

### **2. Test Data Selection Strategy**
For each retraining event, implement **temporal test split**:

```python
# Test Data Selection for Retraining
def create_temporal_test_split(combined_data, test_size=0.2):
    """
    Use the most recent 20% of combined data as test set
    This simulates realistic temporal evaluation pattern
    """
    sorted_data = combined_data.sort_values('timestamp_proxy')  # Use index as time proxy
    split_point = int(len(sorted_data) * 0.8)
    
    train_data = sorted_data.iloc[:split_point]
    test_data = sorted_data.iloc[split_point:]
    
    return train_data, test_data

# Test data evolution:
# Initial training: Random 80/20 split of 20% data
# First retraining: Temporal split of [initial + batch_1-3], test on most recent 20%
# Second retraining: Temporal split of [initial + batch_1-5], test on most recent 20%
```

**Benefits**:
- Realistic evaluation on most recent data patterns
- Tests model's ability to handle latest distribution changes  
- Consistent with healthcare ML best practices for temporal data
- Provides fair comparison across retraining cycles

### **3. Drift Monitoring Visualization Strategy**
Create comprehensive **time-series drift dashboard** showing:

```python
# Visualization Components
drift_timeline = {
    'x_axis': 'batch_processing_time',  # Sequential batch numbers or dates
    'primary_metrics': [
        'dataset_drift_score',      # Overall Evidently drift score (0-1)
        'feature_drift_count',      # Number of features showing significant drift
        'data_quality_score',       # Evidently data quality metrics
        'prediction_drift_score'    # If concept drift detection enabled
    ],
    'secondary_metrics': [
        'model_performance_score',  # F1/accuracy on test data
        'prediction_confidence',    # Model uncertainty metrics
        'data_volume_per_batch'     # Batch size tracking
    ],
    'event_markers': [
        'retraining_events',        # Vertical lines showing when retraining occurred
        'model_deployment_events',  # When new models were deployed  
        'drift_threshold_violations', # Points where thresholds were exceeded
        'synthetic_drift_injections'  # If using synthetic drift for testing
    ]
}

# Multi-panel dashboard layout:
# Panel 1: Drift scores over time with threshold lines
# Panel 2: Model performance metrics over time  
# Panel 3: Feature-level drift heatmap
# Panel 4: Retraining event timeline with trigger reasons
```

**Graph Features**:
- **Time-series line plots** for drift scores with configurable thresholds
- **Event markers** showing retraining points with hover details
- **Performance correlation** showing model accuracy before/after retraining
- **Feature-level breakdown** identifying which features drive drift detection
- **Synthetic drift annotations** for controlled testing scenarios

### **4. Multi-Level Drift Detection Strategy**
```python
# Statistical Methods
drift_methods = {
    'statistical': ['ks_test', 'psi', 'wasserstein_distance'],
    'ml_based': ['drift_detection_trees', 'domain_classifier'], 
    'domain_specific': ['healthcare_metrics', 'seasonal_patterns']
}

# Graduated Response System
drift_responses = {
    'minor_drift': 'log_and_monitor',      # < 0.05 threshold
    'moderate_drift': 'incremental_retrain', # 0.05-0.15 threshold  
    'major_drift': 'full_retrain',         # 0.15-0.30 threshold
    'concept_drift': 'architecture_review'  # > 0.30 threshold
}
```

### **5. Synthetic Drift Injection for Testing**
```python
# Controlled drift scenarios for validation
def inject_covariate_drift(data, features, intensity=0.3):
    """Shift feature distributions to simulate real-world drift"""
    
def inject_concept_drift(data, target_col, drift_type='gradual'):
    """Modify target relationships to test concept drift detection"""
    
def create_seasonal_patterns(data, features, seasonality='monthly'):
    """Add cyclical patterns to test seasonal drift detection"""
```

---

## 📋 **Implementation Roadmap**

## **Phase 5A: Core Implementation** (1-2 weeks)

### **Week 1: Foundation Setup** ✅ **COMPLETED**
- **Step 1**: ✅ Install Evidently AI and drift detection dependencies
- **Step 2**: ✅ Create basic `scripts/monitor_drift.py` with core functionality
- **Step 3**: ✅ Set up S3 paths and MLflow experiment for drift monitoring
- **Step 4**: ✅ Create skeleton monitoring DAG structure

### **Week 2: Basic Integration** ✅ **COMPLETED**  
- **Step 5**: ✅ Implement basic drift detection with Evidently DatasetDriftPreset
- **Step 6**: ✅ Add MLflow logging for drift metrics and reports
- **Step 7**: ✅ Create data batching simulation from future_data.csv
- **Step 8**: ✅ Test end-to-end basic drift detection workflow

## **Phase 5B: Enhanced Features** (2-3 weeks)

### **Week 3: Advanced Drift Detection** ✅ **COMPLETED**
- **Step 9**: ✅ Implement synthetic drift injection functions
- **Step 10**: ✅ Add multiple drift detection methods (KS-test, PSI, Wasserstein)
- **Step 11**: ✅ Create concept drift detection with prediction monitoring
- **Step 12**: ✅ Implement drift visualization dashboard

### **Week 4: Automated Response System** ✅ **COMPLETED**
- **Step 13**: ✅ Modify existing training DAG to handle drift-triggered retraining
- **Step 14**: ✅ Implement data combination logic for retraining
- **Step 15**: ✅ Create graduated response system (minor/moderate/major drift)
- **Step 16**: ✅ Add automatic trigger from monitoring DAG to training DAG

### **Week 5: Monitoring Integration**
- **Step 17**: Create comprehensive monitoring DAG with branching logic
- **Step 18**: Implement batch processing simulation loop
- **Step 19**: Add error handling and recovery mechanisms
- **Step 20**: Test complete drift detection → retraining → deployment loop

## **Phase 5C: Production Polish** (1-2 weeks)

### **Week 6: Dashboard & Alerting**
- **Step 21**: Create MLflow-based drift monitoring dashboard
- **Step 22**: Implement alerting system (email/Slack notifications)
- **Step 23**: Add performance decay monitoring alongside drift detection
- **Step 24**: Create audit trails and compliance logging

### **Week 7: Documentation & Demo**
- **Step 25**: Create comprehensive user documentation
- **Step 26**: Generate sample drift reports and visualizations
- **Step 27**: Record drift detection demonstration video
- **Step 28**: Final testing and system validation

---

## 🛠️ **Detailed Step-by-Step Execution Guide**

### **STEP 1: Install Evidently AI Dependencies**
**Objective**: Add Evidently AI and related drift detection libraries to the system

**Actions**:
1. Update `scripts/requirements-training.txt` to include Evidently AI
2. Update `mlops-services/Dockerfile.airflow` to install drift detection packages
3. Rebuild Airflow services with new dependencies
4. Verify installation in Airflow environment

**Expected Outcome**: Evidently AI available in Airflow workers for drift detection

### **STEP 2: Create Basic Drift Monitoring Script** 
**Objective**: Implement `scripts/monitor_drift.py` with core functionality

**Actions**:
1. Create script structure with argparse for configuration
2. Implement S3 data loading for reference and new batch data
3. Add basic Evidently DatasetDriftPreset integration
4. Implement MLflow logging for drift metrics
5. Add stdout output for Airflow integration

**Expected Outcome**: Functional drift detection script ready for DAG integration

### **STEP 3: Set Up S3 and MLflow Infrastructure**
**Objective**: Configure storage and tracking for drift monitoring

**Actions**:
1. Define S3 paths for drift reports and batch data
2. Create MLflow experiment for drift monitoring
3. Set up directory structure for local drift report generation
4. Configure environment variables for monitoring pipeline

**Expected Outcome**: Infrastructure ready for drift detection workflow

### **STEP 4: Create Monitoring DAG Skeleton**
**Objective**: Build basic Airflow DAG for drift monitoring orchestration

**Actions**:
1. Create `monitoring_retraining_dag.py` with basic structure
2. Implement data batch simulation task
3. Add drift detection task calling monitor_drift.py
4. Create branching logic for drift response
5. Set up XCom passing between tasks

**Expected Outcome**: Orchestration framework ready for drift monitoring

### **STEP 5: Implement Basic Drift Detection**
**Objective**: Create end-to-end drift detection workflow with Evidently DatasetDriftPreset

**Actions**:
1. Test monitor_drift.py script with sample data batches
2. Validate drift detection thresholds and sensitivity
3. Verify data preprocessing consistency between reference and new data
4. Test error handling for edge cases (empty data, missing columns)

**Expected Outcome**: Reliable drift detection on real data samples

### **STEP 6: Add MLflow Logging Integration**
**Objective**: Implement comprehensive experiment tracking for drift monitoring

**Actions**:
1. Verify MLflow experiment logging for drift metrics
2. Test artifact storage for Evidently HTML reports
3. Validate parameter and metric logging consistency
4. Create run naming conventions for drift monitoring

**Expected Outcome**: Complete audit trail for all drift detection runs

### **STEP 7: Create Data Batching Simulation**
**Objective**: Implement realistic data batch simulation from future_data.csv

**Actions**:
1. Create batch splitting logic from future_data.csv
2. Implement configurable batch sizes and intervals
3. Add S3 upload functionality for simulated batches
4. Create batch metadata tracking system

**Expected Outcome**: Automated data batch generation for drift testing

### **STEP 8: Test End-to-End Basic Workflow**
**Objective**: Validate complete basic drift detection pipeline

**Actions**:
1. Run multiple drift detection cycles with different batch sizes
2. Verify S3 storage and MLflow logging across multiple runs
3. Test DAG execution and task dependency handling
4. Debug and refine workflow performance and reliability

**Expected Outcome**: Stable basic drift detection pipeline ready for enhancement

---

## **Phase 5B: Enhanced Features - Detailed Steps**

### **STEP 9: Implement Synthetic Drift Injection** ✅ **COMPLETED**
**Objective**: Create controlled drift scenarios for testing and validation

**Actions**:
1. ✅ Create `inject_covariate_drift()` function for feature distribution shifts
2. ✅ Implement `inject_concept_drift()` for target relationship changes
3. ✅ Add `create_seasonal_patterns()` for cyclical drift simulation
4. ✅ Create drift intensity controls (mild, moderate, severe)
5. ✅ Add drift injection metadata logging for traceability

**Expected Outcome**: ✅ Comprehensive synthetic drift testing capabilities

### **STEP 10: Add Multiple Drift Detection Methods** ✅ **COMPLETED**
**Objective**: Implement advanced statistical drift detection techniques

**Actions**:
1. ✅ Add Kolmogorov-Smirnov test for continuous feature drift
2. ✅ Implement Population Stability Index (PSI) for categorical features
3. ✅ Add Wasserstein distance for distribution comparison
4. ✅ Create ensemble drift scoring combining multiple methods
5. ✅ Add Jensen-Shannon divergence and Chi-square tests

**Expected Outcome**: ✅ Robust multi-method drift detection system

### **STEP 11: Create Concept Drift Detection** ✅ **COMPLETED**
**Objective**: Monitor prediction quality and target relationship changes

**Actions**:
1. ✅ Implement prediction drift monitoring using model outputs
2. ✅ Add performance decay detection over time windows
3. ✅ Create target distribution shift analysis
4. ✅ Implement prediction confidence degradation tracking
5. ✅ Add concept drift severity classification

**Expected Outcome**: ✅ Comprehensive concept drift monitoring capabilities

### **STEP 12: Implement Drift Visualization Dashboard** ✅ **COMPLETED**
**Objective**: Create comprehensive drift monitoring visualization and dashboard

**Actions**:
1. ✅ Create Streamlit-based drift monitoring dashboard
2. ✅ Implement MLflow experiment data loading and visualization
3. ✅ Add real-time drift status metrics display
4. ✅ Create trend analysis charts for drift metrics over time
5. ✅ Add detection methods comparison visualization

**Expected Outcome**: ✅ Interactive drift monitoring dashboard for operational use

### **STEP 13: Modify Training DAG for Drift-Triggered Retraining**
**Objective**: Integrate drift detection with existing training pipeline

**Actions**:
1. Update `training_dag.py` to accept drift trigger parameters
2. Add drift context to training run metadata
3. Implement data combination logic for cumulative retraining
4. Create drift-aware hyperparameter optimization
5. Add retraining success/failure feedback to monitoring system

**Expected Outcome**: Seamless integration between drift detection and model retraining

### **STEP 14: Implement Data Combination Logic**
**Objective**: Create intelligent data merging for drift-triggered retraining

**Actions**:
1. Implement cumulative data combination strategy
2. Create temporal test split functionality for retraining
3. Add data quality validation for combined datasets
4. Implement data balancing and sampling strategies
5. Create data lineage tracking for combined training sets

**Expected Outcome**: Intelligent data management for drift-aware retraining

### **STEP 15: Create Graduated Response System**
**Objective**: Implement intelligent drift response based on severity

**Actions**:
1. Create minor drift response (logging and monitoring)
2. Implement moderate drift response (incremental retraining)
3. Add major drift response (full retraining)
4. Create concept drift response (architecture review alerts)
5. Implement response escalation and de-escalation logic

**Expected Outcome**: Automated drift response system with appropriate severity handling

### **STEP 16: Add Automatic DAG Triggering**
**Objective**: Create seamless automation between monitoring and training DAGs

**Actions**:
1. Implement Airflow DAG triggering from monitoring to training
2. Add parameter passing for drift context and data locations
3. Create trigger validation and error handling
4. Implement trigger rate limiting and safety controls
5. Add trigger audit logging and monitoring

**Expected Outcome**: Fully automated drift detection to retraining pipeline

### **STEP 17: Create Comprehensive Monitoring DAG**
**Objective**: Build production-ready drift monitoring orchestration

**Actions**:
1. Implement advanced DAG structure with parallel processing
2. Add dynamic task generation based on available data batches
3. Create sophisticated branching logic for different drift scenarios
4. Implement task retry and error recovery mechanisms
5. Add comprehensive DAG monitoring and alerting

**Expected Outcome**: Robust production-ready drift monitoring orchestration

### **STEP 18: Implement Batch Processing Simulation Loop**
**Objective**: Create realistic continuous data processing simulation

**Actions**:
1. Implement time-based batch processing simulation
2. Add realistic data arrival patterns and timing
3. Create batch size variation and realistic data scenarios
4. Implement processing backlog and catch-up mechanisms
5. Add batch processing performance monitoring

**Expected Outcome**: Realistic continuous drift monitoring simulation

### **STEP 19: Add Error Handling and Recovery**
**Objective**: Create robust error management for production deployment

**Actions**:
1. Implement comprehensive error classification and handling
2. Add automatic retry mechanisms with exponential backoff
3. Create error notification and escalation procedures
4. Implement graceful degradation for partial failures
5. Add system health monitoring and recovery procedures

**Expected Outcome**: Production-grade error handling and system resilience

### **STEP 20: Test Complete Drift → Retraining → Deployment Loop**
**Objective**: Validate end-to-end automated MLOps workflow

**Actions**:
1. Run complete drift detection to model deployment cycles
2. Test various drift scenarios and response combinations
3. Validate model performance maintenance through drift events
4. Test system performance under load and stress conditions
5. Create comprehensive system validation and acceptance testing

**Expected Outcome**: Fully validated automated drift-aware MLOps pipeline

---

## **Phase 5C: Production Polish - Detailed Steps**

### **STEP 21: Create MLflow-Based Drift Monitoring Dashboard**
**Objective**: Build comprehensive visualization for drift monitoring

**Actions**:
1. Create time-series drift score visualization with threshold lines
2. Implement feature-level drift heatmap with drill-down capabilities
3. Add model performance correlation charts
4. Create retraining event timeline with trigger context
5. Implement real-time dashboard refresh and filtering

**Expected Outcome**: Professional drift monitoring dashboard for stakeholders

### **STEP 22: Implement Alerting System**
**Objective**: Create proactive notification system for drift events

**Actions**:
1. Implement email alerting for drift threshold violations
2. Add Slack/Teams integration for real-time notifications
3. Create escalation procedures for different drift severity levels
4. Implement alert throttling and deduplication
5. Add alert acknowledgment and response tracking

**Expected Outcome**: Comprehensive alerting system for proactive drift management

### **STEP 23: Add Performance Decay Monitoring**
**Objective**: Monitor model accuracy degradation alongside drift detection

**Actions**:
1. Implement model performance tracking over time
2. Add performance decay correlation with drift metrics
3. Create performance threshold alerting
4. Implement prediction confidence monitoring
5. Add business impact assessment for performance changes

**Expected Outcome**: Comprehensive model health monitoring system

### **STEP 24: Create Audit Trails and Compliance Logging**
**Objective**: Implement comprehensive governance and compliance features

**Actions**:
1. Create detailed audit logs for all automated decisions
2. Implement data lineage tracking for drift-triggered retraining
3. Add compliance reporting for regulatory requirements
4. Create decision justification documentation
5. Implement audit trail search and reporting capabilities

**Expected Outcome**: Complete governance and compliance framework

### **STEP 25: Create Comprehensive User Documentation**
**Objective**: Develop complete documentation for system operation

**Actions**:
1. Create drift monitoring system architecture documentation
2. Write operational procedures and troubleshooting guides
3. Develop configuration and customization documentation
4. Create API documentation for integration points
5. Add best practices and optimization guidelines

**Expected Outcome**: Complete documentation suite for system maintainers

### **STEP 26: Generate Sample Drift Reports and Visualizations**
**Objective**: Create demonstration materials and templates

**Actions**:
1. Generate sample drift reports for different scenarios
2. Create visualization templates for common use cases
3. Develop executive summary templates for stakeholders
4. Create training materials for system users
5. Add sample configuration files and examples

**Expected Outcome**: Complete demonstration and training materials

### **STEP 27: Record Drift Detection Demonstration Video**
**Objective**: Create comprehensive system demonstration

**Actions**:
1. Record end-to-end drift detection workflow demonstration
2. Create scenario-based demonstration videos
3. Record configuration and customization tutorials
4. Create troubleshooting and maintenance videos
5. Add executive overview and business value presentations

**Expected Outcome**: Complete video documentation and demonstration suite

### **STEP 28: Final Testing and System Validation**
**Objective**: Complete comprehensive system validation and acceptance

**Actions**:
1. Conduct comprehensive system integration testing
2. Perform load testing and performance validation
3. Execute security and compliance validation
4. Conduct user acceptance testing with stakeholders
5. Create final system validation report and certification

**Expected Outcome**: Production-certified drift monitoring system ready for deployment

---

## 🔧 **Technical Specifications**

### **Dependencies to Add**
```txt
# scripts/requirements-training.txt additions
evidently==0.4.22
scipy>=1.10.0
plotly>=5.14.0
kaleido>=0.2.1
```

### **S3 Structure**
```
s3://health-predict-mlops-f9ac6509/
├── drift_monitoring/
│   ├── batch_data/          # Simulated data batches
│   ├── reports/             # Evidently HTML reports  
│   └── reference_data/      # Reference datasets
└── processed_data/
    └── future_data.csv      # Source for simulation
```

### **MLflow Experiments**
- `HealthPredict_Drift_Monitoring`: Drift detection runs
- `HealthPredict_Training_HPO_Airflow`: Enhanced for drift-triggered retraining

### **Environment Variables**
```bash
# Additional environment variables for drift monitoring
DRIFT_MONITORING_EXPERIMENT=HealthPredict_Drift_Monitoring
DRIFT_REPORTS_S3_PREFIX=drift_monitoring/reports
DRIFT_THRESHOLD_MINOR=0.05
DRIFT_THRESHOLD_MODERATE=0.15
DRIFT_THRESHOLD_MAJOR=0.30
```

---

## 📊 **Success Metrics**

### **Technical Metrics**
- ✅ Drift detection accuracy on synthetic scenarios
- ✅ End-to-end pipeline execution time < 10 minutes
- ✅ Successfully triggered retraining on drift detection
- ✅ Model performance maintained after retraining

### **Operational Metrics**  
- ✅ Zero manual intervention required for drift response
- ✅ Complete audit trail for all automated decisions
- ✅ Alerts delivered within 5 minutes of drift detection
- ✅ 99% uptime for monitoring pipeline

### **Business Metrics**
- ✅ Sustained model F1 score > 0.60 over simulation period
- ✅ Reduced model performance degradation by 80%
- ✅ Automated drift response in < 30 minutes
- ✅ Complete MLOps workflow demonstration ready

---

## 🚨 **Risk Mitigation**

### **Data Quality Risks**
- **Risk**: Poor simulation data quality
- **Mitigation**: Implement multiple synthetic drift scenarios with validation

### **Integration Risks**
- **Risk**: Breaking existing pipeline functionality  
- **Mitigation**: Thorough testing with rollback procedures

### **Performance Risks**
- **Risk**: Drift detection computationally expensive
- **Mitigation**: Optimize for efficiency, implement sampling strategies

### **Operational Risks**
- **Risk**: False positive drift alerts
- **Mitigation**: Implement graduated thresholds and human oversight options

---

## �� **Next Steps**

**Phase 5A: Core Implementation** (Steps 1-8)
- Foundation setup with Evidently AI integration
- Basic drift detection and MLflow logging
- S3 infrastructure and monitoring DAG skeleton
- End-to-end basic workflow validation

**Phase 5B: Enhanced Features** (Steps 9-20)
- Advanced drift detection methods and synthetic injection
- Automated retraining integration with existing pipeline
- Graduated response system and comprehensive monitoring

**Phase 5C: Production Polish** (Steps 21-28)
- Dashboard creation and alerting systems
- Documentation, demonstration, and final validation

---

*For implementation progress and daily updates, see project_work_journal.md* 