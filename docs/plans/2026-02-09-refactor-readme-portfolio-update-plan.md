---
title: Update README to Portfolio-Focused Format
type: refactor
date: 2026-02-09
---

# Update README to Portfolio-Focused Format

## Overview

Replace the current overly-long, technically-dense README.md with an updated version of OLD_README.md. The old README has a better structure for a portfolio piece — business-case driven, concise, and readable. The task is to keep that structure and tone while updating all technical descriptions to match the current architecture.

## Problem Statement

The current README was AI-generated and rewrote the original portfolio-focused document into a technical operations manual. It's too long, too detailed on setup/troubleshooting, and reads like internal documentation rather than a portfolio showcase. The OLD_README.md has the right structure but references outdated architecture (separate DAGs, old metrics, missing features).

## Proposed Solution

Start from OLD_README.md, update technical facts in-place, and write the result to README.md.

## Section-by-Section Changes

### Sections to KEEP AS-IS (no changes needed)
- **Title & hero image** — same
- **Overview** — still accurate
- **Business Problem** — still accurate
- **Key Technologies** — minor updates only (see below)
- **Getting Started > Prerequisites** — still accurate
- **Conclusion** — still accurate
- **Disclaimer note** — still accurate

### Sections to UPDATE

#### 1. Solution Architecture (text + image)

**Keep**: The architecture image `health_predict_high_level_architecture_v2.png` and the general description structure.

**Update bullet points to reflect unified pipeline:**
- OLD: Separate mentions of training pipeline, deployment pipeline, monitoring DAG
- NEW: Single unified DAG `health_predict_continuous_improvement` that handles drift detection, training, quality gates, and deployment
- OLD: "Evidently AI monitors for data drift from S3 logs, triggering automated retraining"
- NEW: "Evidently AI performs drift-gated retraining — KS-test (numeric) and chi-squared (categorical) with a 30% drift threshold controls whether retraining proceeds"

#### 2. Technical Implementation

**Subsection: Data Pipeline & Feature Engineering**
- OLD: Mentions "best-performing model (based on F1 score)"
- NEW: Change to "best-performing model (based on AUC)"
- OLD: Describes Training Phase and Deployment Phase separately
- NEW: Describe as unified pipeline with quality gates

**Subsection: Training Pipeline with HPO**
- OLD: References `training_pipeline_dag.py`
- NEW: Reference `health_predict_continuous_improvement` DAG
- OLD: "Automated selection of best model based on F1 score"
- NEW: "Automated selection of best model based on AUC"
- Keep the training pipeline image

**Subsection: Deployment Pipeline**
- OLD: References `deployment_pipeline_dag.py`
- NEW: Reference deployment phase within the unified DAG
- ADD: "Model version verification" as a deployment step
- ADD: "Regression guardrail" — new model must not degrade AUC by >2%
- Keep the deployment pipeline image

**Subsection: Model Serving API**
- ADD: `/model-info` endpoint for deployment verification
- Otherwise mostly accurate

**Subsection: Drift Detection & Retraining**
- OLD: "PSI, KS-test, and other statistical measures"
- NEW: "KS-test for numeric features, chi-squared for categorical features"
- OLD: "Automated Response: Triggering retraining pipeline when drift exceeds thresholds"
- NEW: "Drift-Gated Retraining: Pipeline only retrains when 30%+ of features show drift; stable data skips retraining"
- ADD: Mention `force_retrain` override capability
- ADD: Mention high-cardinality columns (diag_1/2/3) excluded from drift analysis

#### 3. Key Technologies

- ADD: **GitHub Actions** under a CI/CD bullet
- ADD: **Streamlit** under monitoring/dashboard
- ADD: **Evidently AI** (already there, just confirm)
- Everything else stays

#### 4. Results & Achievements

- OLD: "85%+ F1 score" — this is inaccurate
- NEW: "64%+ AUC" with note about realistic performance on imbalanced healthcare data
- OLD: "70% reduction in time to deploy" — unsubstantiated claim
- NEW: "100% automated pipeline from data ingestion to production deployment"
- OLD: "99.9% uptime" — unsubstantiated
- NEW: "Zero-downtime deployments with Kubernetes rolling updates"
- Keep: Monitoring, Scalability, Compliance bullets (update wording to be accurate)

#### 5. Getting Started > Quick Start

- OLD: References separate DAGs (`health_predict_training_hpo`, `health_predict_api_deployment`, `monitoring_retraining_dag`)
- NEW: Single DAG `health_predict_continuous_improvement` with `batch_number` parameter
- OLD: Manual docker-compose startup
- NEW: Reference `./scripts/start-mlops-services.sh` as primary startup method
- ADD: Mention Streamlit dashboard at port 8501

#### 6. Future Enhancements

- REMOVE: "Multi-Cloud Support" (not realistic next step)
- KEEP: A/B Testing, Explainability (SHAP/LIME)
- UPDATE: Add more relevant enhancements based on current system:
  - Advanced drift detection metrics (PSI, Wasserstein distance)
  - Real-time streaming inference (Kafka)
  - Feature store integration (Feast)
  - Automated rollback on regression
- Keep this section concise (5-6 bullet points, not detailed paragraphs)

### Sections to ADD (briefly)

#### CI/CD badge
Add GitHub Actions CI/CD badge at the top (already in current README):
```
![CI/CD Pipeline](https://github.com/bencappello/health-predict/actions/workflows/ci-cd.yml/badge.svg)
```

#### Monitoring Dashboard (1-2 sentences)
Brief mention of Streamlit dashboard under the architecture or as a small subsection. Don't overdo it — just note it exists at port 8501 for drift trends, model performance, and API health.

#### Dual-Pipeline CI/CD (1-2 sentences)
Brief mention that the system uses GitHub Actions for software CI/CD and Airflow for ML CI/CD. One sentence, not a whole section.

### Sections to NOT ADD (keep it concise)
- NO detailed project structure tree
- NO API documentation with request/response examples
- NO troubleshooting guide
- NO mermaid diagrams (keep the PNG images)
- NO drift-aware batch profiles table
- NO service management commands
- NO model performance metrics table

## Acceptance Criteria

- [x] README.md matches OLD_README.md structure and length (~160 lines, not 680) — 172 lines
- [x] All 4 images retained: `readme_hero.png`, `health_predict_high_level_architecture_v2.png`, `training_pipeline.png`, `deployment_pipeline.png`
- [x] DAG references updated to `health_predict_continuous_improvement`
- [x] Drift detection accurately described (KS-test + chi-squared, 30% threshold, gating)
- [x] Model selection metric changed from F1 to AUC
- [x] Results section has realistic, accurate claims
- [x] Quick Start references startup script and correct DAG name
- [x] CI/CD badge added
- [x] Streamlit dashboard mentioned
- [x] GitHub Actions CI/CD mentioned
- [x] Quality gates mentioned (drift gate, regression guardrail, deployment verification)
- [x] Tone reads as portfolio piece, not operations manual
- [x] OLD_README.md preserved (not modified or deleted)

## Implementation

Single file change: `README.md`

**Approach**: Read OLD_README.md, make the updates listed above in-place, write to README.md. One pass, one file.

## References

- Source: `OLD_README.md` (portfolio structure to preserve)
- Current: `README.md` (facts to extract, structure to discard)
- Architecture: `CLAUDE.md` (authoritative source of truth for current system)
- Images: `images/` directory (4 PNG files to retain)
