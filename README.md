# Health Predict: End-to-End MLOps System for Patient Readmission Prediction

![CI/CD Pipeline](https://github.com/bencappello/health-predict/actions/workflows/ci-cd.yml/badge.svg)

<img src="images/readme_hero.png" alt="Diagram" width="500"/>

## Overview

**Health Predict** is a comprehensive MLOps system designed to predict the risk of patient readmission in healthcare settings. This project implements a complete machine learning lifecycle from data ingestion to automated retraining, with a focus on robust MLOps practices and production-grade deployment.

This system addresses the critical healthcare challenge of patient readmission, which not only burdens healthcare systems financially but also indicates potential gaps in patient care quality. By accurately predicting readmission risk, healthcare providers can implement targeted interventions, improving patient outcomes while optimizing resource allocation.

## Business Problem

Patient readmission—the return of a patient to the hospital shortly after discharge—represents both a significant healthcare quality issue and a financial burden:

- Hospital readmissions cost the U.S. healthcare system approximately $26 billion annually
- Readmissions often indicate unresolved health issues or inadequate post-discharge support
- CMS penalizes hospitals for excessive readmission rates through the Hospital Readmissions Reduction Program
- Early identification of at-risk patients enables proactive interventions

Health Predict addresses this challenge by leveraging machine learning to predict which patients are most likely to be readmitted, allowing for targeted interventions before discharge and during follow-up care.

## Solution Architecture

The Health Predict system implements a complete MLOps lifecycle on AWS infrastructure, employing cost-effective design choices to demonstrate enterprise-level capabilities while maintaining budget efficiency:

*   **Data Pipeline**: Patient data is ingested into S3 and processed by a unified pipeline orchestrated by Airflow on EC2.
*   **Model Development**: Ray Tune handles distributed hyperparameter optimization across multiple model types (XGBoost, Random Forest, Logistic Regression), with experiments and models managed by MLflow.
*   **Quality Gates**: Drift-gated retraining ensures the pipeline only retrains when data distribution changes are detected. A regression guardrail prevents deploying models that degrade performance.
*   **Deployment**: Production models are containerized, stored in ECR, and deployed to a Kubernetes (Minikube) cluster with rolling updates and automated version verification.
*   **Model Serving**: A FastAPI service serves predictions to end-users, with health checks and model metadata endpoints.
*   **Monitoring & Retraining**: Evidently AI performs drift-gated retraining — using KS-test (numeric) and chi-squared (categorical) statistical tests with a 30% drift threshold to control whether retraining proceeds. A Streamlit dashboard provides real-time visibility into drift trends, model performance, and deployment status.

![Health Predict Solution Architecture](images/health_predict_high_level_architecture_v2.png)

## Technical Implementation

### Unified Training & Deployment Pipeline

The entire ML lifecycle is managed by a single Airflow DAG (`health_predict_continuous_improvement`) that orchestrates everything from drift detection through production deployment:

*   **Drift Detection**: Evidently AI analyzes ~44 features (11 numeric via KS-test, 33 categorical via chi-squared) against the training distribution. High-cardinality diagnostic codes are excluded to reduce noise. If fewer than 30% of features show drift, retraining is skipped.
*   **Training Phase**: When drift is detected, data from all previous batches is combined (cumulative learning), processed through feature engineering, and used by Ray Tune for hyperparameter optimization. MLflow tracks experiments and registers the best-performing model (based on AUC).
*   **Quality Gate**: A regression guardrail compares the new model against the current production model — deployment is blocked if AUC degrades by more than 2%.
*   **Deployment Phase**: The production model is packaged into a Docker container with the FastAPI application, pushed to ECR, and deployed to Kubernetes via a rolling update. Automated verification confirms the deployed model version matches what was promoted.

### Training Pipeline with Hyperparameter Optimization

The training pipeline leverages distributed computing for efficient model development:

- **Orchestration**: Single Airflow DAG manages the end-to-end process
- **Models**: Multiple model types (XGBoost, Random Forest, Logistic Regression)
- **HPO**: Ray Tune with ASHA scheduler for scalable hyperparameter optimization with early stopping
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Model Selection**: Automated selection of best model based on AUC
- **Model Registry**: Automatic promotion to production in MLflow registry

![Training Pipeline](images/training_pipeline.png)

### Deployment Pipeline

The deployment pipeline automates the transition from model training to production:

- **Regression Guardrail**: New model must not degrade AUC by more than 2% vs. production
- **Container Building**: Automated Docker image building with the latest model
- **Registry**: Pushing versioned images to Amazon ECR
- **Kubernetes Deployment**: Rolling updates for zero-downtime deployments
- **Version Verification**: Automated check that deployed API serves the correct model version
- **Health Monitoring**: Readiness and liveness probes on the prediction API

![Deployment Pipeline](images/deployment_pipeline.png)

### Model Serving API

The model serving API provides reliable and efficient prediction services:

- **Framework**: FastAPI for high-performance REST API
- **Documentation**: Auto-generated Swagger/OpenAPI docs
- **Model Loading**: Dynamic loading from MLflow registry
- **Preprocessing**: Consistent preprocessing pipeline with the training process
- **Model Metadata**: `/model-info` endpoint for deployment verification
- **Health Checks**: Endpoint for K8s readiness/liveness probes

### Drift Detection & Retraining

The monitoring system ensures model performance over time:

- **Data Partitioning**: 20% of data for initial training, 80% reserved to simulate future data across 5 batches with intentional drift profiles
- **Drift Detection**: Evidently AI with KS-test for numeric features and chi-squared for categorical features
- **Drift-Gated Retraining**: Pipeline only retrains when 30%+ of features show drift; stable data skips retraining entirely
- **Override**: `force_retrain` parameter available to bypass the drift gate when needed
- **Visualization**: Streamlit dashboard for tracking drift metrics, model performance, and deployment status

### CI/CD Architecture

The system uses a dual-pipeline approach: **GitHub Actions** handles software CI/CD (test, build, push, deploy on code changes), while **Airflow** handles ML CI/CD (drift detection, retraining, model promotion on data events). Both pipelines can deploy independently to the same Kubernetes cluster.

## Key Technologies

The project leverages a diverse technology stack:

- **AWS Services**: EC2, S3, ECR
- **Containerization**: Docker, Docker Compose, Kubernetes (Minikube)
- **MLOps Tools**: MLflow, Apache Airflow, Evidently AI
- **Machine Learning**: Scikit-learn, XGBoost, Ray Tune
- **API Development**: FastAPI, Pydantic
- **Data Processing**: Pandas, NumPy
- **Monitoring**: Streamlit
- **CI/CD**: GitHub Actions
- **Infrastructure as Code**: Terraform

## Results & Achievements

- **Performance**: Models achieve 64%+ AUC in predicting patient readmission on highly imbalanced healthcare data
- **Automation**: 100% automated pipeline from data ingestion to production deployment
- **Intelligence**: Drift-gated retraining prevents unnecessary model updates when data is stable
- **Reliability**: Zero-downtime deployments with Kubernetes rolling updates and automated version verification
- **Monitoring**: Real-time drift detection and model performance tracking via Streamlit dashboard
- **Compliance**: Complete model lineage tracking and reproducibility through MLflow

## Getting Started

### Prerequisites

- AWS Account with appropriate permissions
- Docker and Docker Compose installed
- Terraform installed (for infrastructure provisioning)
- Minikube installed (for local Kubernetes deployment)

### Quick Start

1. **Infrastructure Setup**
   ```bash
   cd iac
   terraform init
   terraform apply
   ```

2. **MLOps Services Startup**
   ```bash
   ./scripts/start-mlops-services.sh
   ```

3. **Run the Pipeline**
   - Access Airflow UI at http://<EC2_PUBLIC_IP>:8080
   - Trigger the `health_predict_continuous_improvement` DAG with config: `{"batch_number": 1}`
   - The pipeline handles drift detection, training, quality gates, and deployment automatically

4. **Monitor & Access**
   - Prediction API at http://<MINIKUBE_IP>:<NODE_PORT>
   - MLflow UI at http://<EC2_PUBLIC_IP>:5000
   - Monitoring Dashboard at http://<EC2_PUBLIC_IP>:8501

## Future Enhancements

- **A/B Testing**: Implement automated comparison of model versions in production using traffic splitting (e.g., Istio weighted routing or Kubernetes canary deployments). This would allow new models to serve a percentage of live traffic while monitoring real-world performance metrics before full rollout, reducing the risk of deploying a model that performs well on test data but poorly on live patient encounters.

- **Explainability**: Integrate SHAP (SHapley Additive exPlanations) or LIME to provide per-prediction feature importance explanations. In clinical settings, physicians need to understand *why* a patient is flagged as high-risk for readmission — not just the probability score. SHAP values could be computed at prediction time and returned alongside the risk score via the API, with aggregate feature importance tracked across batches in the monitoring dashboard.

- **Advanced Drift Metrics**: Supplement the current KS-test and chi-squared drift detection with Population Stability Index (PSI) and Wasserstein distance. PSI provides a standardized drift severity score that is more interpretable for stakeholders, while Wasserstein distance captures distribution shape changes that KS-test may miss. These metrics could be added to the existing Evidently-based detection pipeline and visualized in the Streamlit dashboard alongside the current drift share metric.

- **Real-time Streaming**: Replace the current batch-triggered pipeline with a Kafka-based streaming architecture that processes patient data as encounters are completed. This would enable near-real-time drift detection and continuous model updates rather than the current batch-interval approach. Implementation would involve a Kafka producer at the hospital EHR interface, a Flink or Spark Streaming consumer for feature engineering, and a modified Airflow DAG that triggers on stream-detected drift events.

- **Feature Store**: Adopt a dedicated feature store such as Feast to centralize feature definitions, ensure consistency between training and serving pipelines, and enable feature versioning. Currently, the same feature engineering code is shared between training and inference via Python imports, but a feature store would add point-in-time correctness, feature lineage tracking, and the ability to reuse features across multiple models or teams.

- **Automated Rollback**: Extend the post-deployment health check to continuously monitor prediction API performance (latency, error rates, prediction distribution shifts) and automatically trigger a rollback to the previous model version if degradation is detected. This would use the existing MLflow model registry's version history to revert deployments, combined with Kubernetes rollback commands, providing a safety net beyond the current pre-deployment quality gate.

## Conclusion

Health Predict demonstrates a comprehensive MLOps approach to healthcare predictive modeling, showcasing the entire machine learning lifecycle from data preparation to production monitoring and automated retraining. The system balances sophisticated ML capabilities with cost-effective infrastructure choices, making it both powerful and practical for real-world healthcare applications.

This project highlights the critical intersection of machine learning and healthcare, providing a tool that not only predicts patient outcomes but does so within a robust framework that ensures reliability, scalability, and adaptability over time.
