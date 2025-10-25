# Step 03: API Development (Phase 3)

Time: 6–8 hours

Sub-steps:
- 3.1 Prediction Service (2–3 hours)
- 3.2 Feedback Service (2 hours)
- 3.3 Testing & Documentation (2–3 hours)

---

## 3.1 Prediction Service (2–3 hours)

Tasks
- Create BentoML service class
- Implement model loading from MLflow
- Create /predict endpoint with input validation
- Add Prometheus metrics (request count, latency)
- Implement hot-swap mechanism (watch for new model versions)
- Create bentofile.yaml for packaging

Deliverables
- services/prediction/service.py — BentoML service
- services/prediction/bentofile.yaml — service configuration
- API documentation (endpoint specs)

How to Validate
- Ensure MLflow server is running
- Start service: bentoml serve services/prediction/service.py:PredictionService
- Service should start on port 3000
- Test with curl or Python requests:
  - POST /predict with sensor data
  - Should return JSON with rul_prediction, model_version
  - Invalid input should return appropriate error
- Check /metrics endpoint returns Prometheus format metrics
- Test model hot-swap: save new model to MLflow, verify service picks it up (may need restart for MVP)
- Verify service handles multiple concurrent requests

---

## 3.2 Feedback Service (2 hours)

Tasks
- Create BentoML feedback service
- Implement /feedback endpoint for collecting actual RUL values
- Store feedback data (JSON file or in-memory for MVP)
- Add Prometheus metrics
- Create bentofile.yaml

Deliverables
- services/feedback/service.py — feedback service
- services/feedback/bentofile.yaml
- Feedback storage mechanism

How to Validate
- Start service: bentoml serve services/feedback/service.py:FeedbackService --port 3001
- Test with curl/requests:
  - POST /feedback with prediction and actual RUL
  - Should return success confirmation
  - Data should be stored (check file or database)
- Check /metrics endpoint works
- Send multiple feedback entries and verify all are stored
- Verify stored data format is correct for later drift detection use

---

## 3.3 Testing & Documentation (2–3 hours)

Tasks
- Write unit tests for both services
- Create integration tests
- Test error handling (invalid inputs, missing data)
- Document API endpoints with examples
- Create example curl commands

Deliverables
- tests/test_apis.py — API test suite
- API documentation (README or separate doc)
- Example requests file

How to Validate
- Run: pytest tests/test_apis.py -v — all tests pass
- Test suite should cover:
  - Valid prediction requests
  - Invalid input handling
  - Feedback storage
  - Metrics endpoints
  - Concurrent requests
- All API endpoints documented with example requests
- Example curl commands work when copied directly
