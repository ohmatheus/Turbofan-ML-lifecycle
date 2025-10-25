# Predictive Maintenance MLOps Plan

This folder reorganizes the original Implementation Guide into a clean, step‑by‑step plan from Step 1 to Step 10. The aim is to preserve all information while making the execution order obvious and easy to follow.

Use the table of contents below to navigate to each step. The main general information about the project (overview, architecture, tech stack, etc.) is documented on this page.

https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
---

## Table of Contents
- Step 01: Foundation & Data → ./step-01-foundation-and-data.md
- Step 02: Model Development → ./step-02-model-development.md
- Step 03: API Development → ./step-03-api-development.md
- Step 04: Monitoring Stack → ./step-04-monitoring-stack.md
- Step 05: Drift Detection & Automation → ./step-05-drift-detection-and-automation.md
- Step 06: Containerization → ./step-06-containerization.md
- Step 07: Production Simulation & Demo → ./step-07-production-simulation.md
- Step 08: Documentation & Code Quality → ./step-08-documentation.md
- Step 09 (Optional): Kubernetes Deployment → ./step-09-kubernetes-deployment.md
- Step 10 (Optional): Advanced Deep Learning → ./step-10-advanced-deep-learning.md

---

## Project Overview
- Project Name: Predictive Maintenance for Industrial Equipment - Full ML Lifecycle
- Objective: Build a complete, production-grade ML system demonstrating the entire ML lifecycle including training, deployment, monitoring, automated drift detection, and retraining.
- Problem Statement: Predict Remaining Useful Life (RUL) and failure probability for industrial turbofan engines using sensor data, with continuous monitoring and automated model updates.

---

## Context & Constraints

### Project Requirements
- Full ML Lifecycle: Training → Deployment → Monitoring → Drift Detection → Retraining
- Real-world Data: NASA Turbofan Engine Degradation Dataset from Kaggle
- Production Architecture: Multiple microservices with proper API design
- Monitoring Stack: Prometheus + Grafana for metrics visualization
- API Framework: BentoML for ML model serving
- Containerization: Docker + Docker Compose (Kubernetes optional)
- Experiment Tracking: MLflow for model versioning
- Cost: $0 - Everything runs locally
- Repository: Single monorepo structure
- Automation: Drift detection triggers automatic retraining

### Technical Constraints
- Must run locally on a single machine
- Docker Compose for orchestration (Kubernetes optional)
- No browser storage APIs (localStorage/sessionStorage)
- Time budget: 34-48 hours for MVP, +6-10h for K8s, +12-18h for deep learning
- Complexity: Medium - production-ready but not over-engineered

### Dataset Details: NASA Turbofan Engine Degradation Dataset
- Multiple engine units (100+ engines)
- Sequential time-series data (cycles until failure)
- 21 sensor readings per cycle (temperature, pressure, vibration, etc.)
- 3 operational settings
- Natural drift between engine batches

### Drift Strategy (Subset quarter simulation)
- Simulate time as four equal quarters: T0, T1, T2, T3
- Map C-MAPSS subsets sequentially as quarters: T0 → FD001, T1 → FD002, T2 → FD003, T3 → FD004
- Training/Baseline: Train on FD001 and compute baseline stats from FD001
- Validation: Validate on FD002 with PSI expected to rise slightly but stay below threshold
- Production Simulation: Stream quarters in order FD001 → FD002 → FD003 → FD004
- Quarter Boundary Mixing: At each boundary (FD001→FD002, FD002→FD003, FD003→FD004), sporadically inject a small batch of engine units from the upcoming subset into the current production stream to create progressive drift
- Automation: When drift exceeds thresholds, automatically trigger retraining and redeploy the new model
- Runtime: Keep demo runtime around 10 minutes using light models and sampling

---

## System Architecture

High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single Repository                             │
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Training        │         │  Production      │             │
│  │  Pipeline        │────────▶│  Simulator       │             │
│  │  (MLflow)        │         │  (Batch Stream)  │             │
│  └──────────────────┘         └──────────────────┘             │
│         │                              │                         │
│         │ Saves Models                 │ Sends Requests          │
│         ▼                              ▼                         │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Model Registry  │◀────────│  Prediction API  │             │
│  │  (MLflow)        │         │  (BentoML)       │             │
│  └──────────────────┘         └──────────────────┘             │
│         ▲                              │                         │
│         │                              ▼                         │
│         │                      ┌──────────────────┐             │
│         │                      │  Feedback API    │             │
│         │                      │  (BentoML)       │             │
│         │                      └──────────────────┘             │
│         │                              │                         │
│         │                              ▼                         │
│         │                      ┌──────────────────┐             │
│         │                      │  Prometheus      │             │
│         │                      │  (Metrics)       │             │
│         │                      └──────────────────┘             │
│         │                              │                         │
│         │                              ▼                         │
│         │                      ┌──────────────────┐             │
│         │                      │  Grafana         │             │
│         │                      │  (Dashboards)    │             │
│         │                      └──────────────────┘             │
│         │                              │                         │
│         │                              ▼                         │
│         │                      ┌──────────────────┐             │
│         │                      │  Drift Detector  │             │
│         │                      │  (Monitoring)    │             │
│         │                      └──────────────────┘             │
│         │                              │                         │
│         │                              │ Triggers                │
│         │                              ▼                         │
│         │                      ┌──────────────────┐             │
│         └──────────────────────│  Retraining      │             │
│                                │  Service         │             │
│                                │  (BentoML)       │             │
│                                └──────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Microservices Components
1. Prediction Service (BentoML) - Inference API with hot-swappable models
2. Feedback Service (BentoML) - Collects actual values for monitoring
3. Drift Detector Service - Scheduled monitoring (every 10 min)
4. Retraining Service (BentoML) - Automated model retraining
5. MLflow Tracking Server - Experiment tracking and model registry
6. Prometheus - Metrics collection and storage
7. Grafana - Visualization dashboards
8. Production Simulator - Streams engine data for testing

---

## Repository Structure
```
turbofan-predictive-maintenance/
├── README.md
├── docker-compose.yml
├── requirements.txt
│
├── data/
│   ├── raw/                          # NASA dataset
│   ├── processed/                    # Preprocessed data
│   └── baseline_stats.json           # Baseline metrics
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── model_utils.py
│   ├── drift/
│   │   ├── detector.py
│   │   ├── metrics.py
│   │   └── thresholds.py
│   └── utils/
│       ├── config.py
│       └── prometheus_metrics.py
│
├── services/
│   ├── prediction/
│   │   ├── service.py
│   │   ├── bentofile.yaml
│   │   └── Dockerfile
│   ├── feedback/
│   │   ├── service.py
│   │   ├── bentofile.yaml
│   │   └── Dockerfile
│   ├── retraining/
│   │   ├── service.py
│   │   ├── bentofile.yaml
│   │   └── Dockerfile
│   └── drift_detector/
│       ├── detector_service.py
│       └── Dockerfile
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
│
├── simulation/
│   ├── production_simulator.py
│   └── drift_scenarios.py
│
└── tests/
    ├── test_data_processing.py
    ├── test_model.py
    ├── test_drift_detection.py
    └── test_apis.py
```

---

## Technology Stack

### Core Technologies
- Python: 3.9+
- ML Framework: Scikit-learn (Random Forest for MVP), TensorFlow/Keras (Phase 10)
- API Framework: BentoML 1.2+
- Experiment Tracking: MLflow 2.0+
- Monitoring: Prometheus + Grafana
- Containerization: Docker + Docker Compose
- Data Processing: Pandas, NumPy
- Scheduling: APScheduler

### Key Libraries
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
mlflow>=2.0.0
bentoml>=1.2.0
prometheus-client>=0.16.0
requests>=2.28.0
pyyaml>=6.0
APScheduler>=3.10.0
scipy>=1.9.0
tensorflow>=2.12.0  # For Phase 10
optuna>=3.0.0       # For Phase 10
shap>=0.41.0        # For Phase 10
```

---

## Complete Project Timeline

| Phase | Description | Time | Cumulative |
|-------|-------------|------|------------|
| 1 | Foundation & Data | 4-6h | 4-6h |
| 2 | Model Development | 6-8h | 10-14h |
| 3 | API Development | 6-8h | 16-22h |
| 4 | Monitoring Stack | 4-6h | 20-28h |
| 5 | Drift & Automation | 6-8h | 26-36h |
| 6 | Containerization | 3-5h | 29-41h |
| 7 | Simulation & Demo | 3-4h | 32-45h |
| 8 | Documentation | 2-3h | 34-48h |
| Total (MVP) | Without K8s/DL | 34-48h | - |
| 9 (Optional) | Kubernetes Deployment | 6-10h | 40-58h |
| 10 (Optional) | Deep Learning | 12-18h | 46-66h |

---

## Validation Commands (Quick Health Checks)
- docker-compose ps
- Prediction API test (POST /predict)
- Feedback API test (POST /feedback)
- Prometheus metrics: curl http://localhost:3000/metrics and :3001/metrics
- Prometheus targets UI: http://localhost:9090/targets
- MLflow UI: http://localhost:5001
- Grafana UI: http://localhost:3002 (admin/admin)
- Docker logs: docker-compose logs -f <service>
- Kubernetes (if using K8s): kubectl get pods/services/logs/describe
- Run tests: pytest tests/ -v

Full, detailed validation commands remain intact within each relevant step file.

---

## Final Checklist

### Before Phase Completion
- [ ] All tasks in phase completed
- [ ] All deliverables created and in correct location
- [ ] Validation steps executed successfully
- [ ] Code tested and working
- [ ] Documentation updated
- [ ] Git commit with clear message

### Before Project Completion

Functionality
- [ ] Model trains successfully with RMSE < 20 (RF) or < 15 (LSTM)
- [ ] Prediction API responds in < 100ms (p95)
- [ ] Feedback API stores data correctly
- [ ] Drift detection calculates PSI and RMSE
- [ ] Retraining triggers automatically when drift detected
- [ ] New model deploys with hot-swap
- [ ] Full workflow works end-to-end

Infrastructure
- [ ] All services start with docker-compose up
- [ ] Services communicate correctly
- [ ] MLflow tracks all experiments
- [ ] Prometheus scrapes all metrics
- [ ] Grafana displays real-time dashboards
- [ ] Production simulator streams data

Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] API tests pass
- [ ] Drift detection tests pass
- [ ] End-to-end demo works

Documentation
- [ ] README is comprehensive and clear
- [ ] Architecture diagram included
- [ ] Setup instructions complete
- [ ] API endpoints documented
- [ ] Troubleshooting guide exists
- [ ] All code has docstrings

Code Quality
- [ ] Code formatted consistently
- [ ] No commented-out code
- [ ] No hardcoded secrets
- [ ] Proper error handling
- [ ] Type hints where appropriate
- [ ] .gitignore configured

Repository
- [ ] Clean git history
- [ ] No large files in repo
- [ ] No data files in repo
- [ ] README at root
- [ ] License file included
- [ ] Requirements.txt complete
