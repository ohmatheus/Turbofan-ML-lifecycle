# Step 02: Model Development (Phase 2)

Time: 6–8 hours

Sub-steps:
- 2.1 Data Preprocessing (2 hours)
- 2.2 Feature Engineering (2–3 hours)
- 2.3 Model Training (2–3 hours)

---

## 2.1 Data Preprocessing (2 hours)

Tasks
- Create data loading module
- Add column names to raw data
- Calculate RUL (Remaining Useful Life) for each row
- Normalize/scale sensor features
- Prepare subset-aware processing for C-MAPSS FD001–FD004
- Split by subsets: Train on FD001, Validate on FD002, Reserve FD003–FD004 for production simulation
- Save processed datasets

Deliverables
- src/data/load_data.py — loads raw data with proper columns
- src/data/preprocess.py — preprocessing pipeline
- Processed CSV files: data/processed/train.csv, val.csv, test.csv

How to Validate
- Run: python src/data/preprocess.py — should complete without errors
- Check that processed CSV files exist in data/processed/
- Load train.csv and verify:
  - Has RUL column with values from 0 to max_cycle
  - All sensor columns are normalized (values roughly between 0 and 1 or -1 and 1)
  - No missing values (NaN)
  - Train set has ~60 unique engine units
  - Val set has ~20 unique engine units
- Print shape and first few rows to confirm structure

---

## 2.2 Feature Engineering (2–3 hours)

Tasks
- Create rolling window features (moving averages for sensors)
- Calculate sensor deltas (rate of change)
- Add time-based features (cycle normalization)
- Create interaction features between operating settings
- Save feature-engineered datasets

Deliverables
- src/data/feature_engineering.py — feature creation functions
- notebooks/02_feature_engineering.ipynb — experimentation
- Enhanced datasets: data/processed/train_features.csv, val_features.csv

How to Validate
- Run: python src/data/feature_engineering.py — completes successfully
- Load train_features.csv and verify:
  - Original columns + new feature columns present
  - Rolling mean features exist (e.g., sensor_1_rolling_mean_5)
  - Delta features exist (e.g., sensor_1_delta)
  - No NaN values (rolling windows should be handled with fillna)
  - Total columns > original (should have 50–100+ features)
- Check feature correlation with RUL to identify most predictive features
- Document top 10 most important features

---

## 2.3 Model Training (2–3 hours)

Tasks
- Set up MLflow tracking server
- Create training script with Random Forest model
- Log parameters, metrics, and model to MLflow
- Train on train_features.csv, evaluate on val_features.csv
- Save best model to MLflow registry
- Document baseline metrics

Deliverables
- src/models/train.py — training script with MLflow integration
- src/models/evaluate.py — evaluation utilities
- Trained model saved in MLflow with metrics
- Baseline performance document

How to Validate
- Start MLflow server: mlflow server --host 0.0.0.0 --port 5001
- Run training: python src/models/train.py
- Open MLflow UI at http://localhost:5001
- Verify:
  - Experiment exists with runs
  - Latest run shows logged parameters (n_estimators, max_depth, etc.)
  - Metrics are logged (RMSE, MAE, R²)
  - RMSE should be < 25 for reasonable baseline
  - Model artifact is saved
- Load model from MLflow and make test prediction to confirm it works
- Document metrics in README or separate file
