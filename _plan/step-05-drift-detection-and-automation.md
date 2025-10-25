# Step 05: Drift Detection & Automation (Phase 5)

Time: 6–8 hours

Sub-steps:
- 5.1 Drift Detection Logic (2–3 hours)
- 5.2 Drift Detector Service (2–3 hours)
- 5.3 Retraining Service (2–3 hours)
- 5.4 Hot-Swap Implementation (1 hour)

---

## 5.1 Drift Detection Logic (2–3 hours)

Tasks
- Implement PSI (Population Stability Index) calculation
- Implement KS-test for distribution comparison
- Calculate performance metrics (RMSE over time)
- Define drift thresholds
- Create baseline statistics from FD001 (training) subset

Deliverables
- src/drift/metrics.py — drift calculation functions
- src/drift/thresholds.py — threshold definitions
- data/baseline_stats.json — baseline feature distributions

How to Validate
- Create test script that calculates PSI between two datasets
- Use FD001 as baseline (training subset)
- Use FD002 as first comparison (expected mild drift, PSI generally < threshold)
- Optionally compare FD003 and FD004 against FD001 to observe progressively higher PSI
- PSI should be calculated for each sensor feature
- Test with modified data (add noise) — PSI should increase
- Calculate RMSE on FD002 validation set, save as baseline
- Create baseline_stats.json with feature means, stds, percentiles

---

## 5.2 Drift Detector Service (2–3 hours)

Tasks
- Create scheduled monitoring service using APScheduler
- Fetch recent predictions and feedback data
- Calculate drift metrics (PSI, performance)
- Expose metrics to Prometheus
- Trigger retraining API when drift detected

Deliverables
- services/drift_detector/detector_service.py
- Dockerfile for drift detector

How to Validate
- Run drift detector service
- Should check for drift every 10 minutes (or configured interval)
- Verify it can:
  - Fetch feedback data from feedback service
  - Calculate PSI for recent data vs baseline
  - Calculate current RMSE
  - Expose drift metrics to Prometheus
- Manually trigger drift check
- Check logs show PSI scores and RMSE
- When drift threshold exceeded, verify retraining trigger is called (can stub for now)

---

## 5.3 Retraining Service (2–3 hours)

Tasks
- Create BentoML retraining service
- Implement /retrain endpoint
- Fetch latest production data (feedback)
- Combine with training data
- Train new model
- Validate new model performance
- Save to MLflow if better
- Update baseline statistics

Deliverables
- services/retraining/service.py
- services/retraining/bentofile.yaml

How to Validate
- Start retraining service
- Manually trigger retraining: POST to /retrain
- Service should:
  - Fetch feedback data
  - Load original training data
  - Combine datasets
  - Train new model
  - Evaluate on validation set
  - Compare RMSE to current baseline
  - Save to MLflow if RMSE ≤ baseline * 1.1
- Check MLflow for new model version
- Verify model version incremented
- Check logs show training progress and validation results
- Trigger retraining with insufficient data — should handle gracefully

---

## 5.4 Hot-Swap Implementation (1 hour)

Tasks
- Add model version checking to prediction service
- Implement background thread to poll MLflow for new versions
- Reload model when new version detected
- Ensure zero-downtime during swap

Deliverables
- Updated prediction service with hot-swap capability

How to Validate
- Start prediction service with model v1
- Make predictions — note model version in response
- Train and save new model (v2) to MLflow
- Wait for hot-swap interval (e.g., 60 seconds)
- Make new prediction — should show model v2
- Verify no service downtime during swap
- Check logs show model reload message
- Test that predictions still work correctly with new model
