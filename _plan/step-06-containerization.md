# Step 06: Containerization (Phase 6)

Time: 3–5 hours

Sub-steps:
- 6.1 Docker Compose Configuration (2–3 hours)
- 6.2 Integration Testing (1–2 hours)

---

## 6.1 Docker Compose Configuration (2–3 hours)

Tasks
- Create Dockerfile for each service (prediction, feedback, retraining, drift-detector)
- Write docker-compose.yml for all services
- Configure service networking
- Set up volume mounts for data and models
- Configure environment variables

Deliverables
- docker-compose.yml — full stack definition
- services/*/Dockerfile — one per service
- .dockerignore files

How to Validate
- Run: docker-compose up --build
- All services should start:
  - prediction-api
  - feedback-api
  - retraining-service
  - drift-detector
  - mlflow
  - prometheus
  - grafana
- Check docker-compose ps — all should show "Up"
- Test inter-service communication:
  - Prediction API can access MLflow
  - Drift detector can access feedback API
  - Prometheus can scrape all services
- Make prediction through Docker network
- Check Grafana dashboards show data

---

## 6.2 Integration Testing (1–2 hours)

Tasks
- Test full end-to-end flow with Docker Compose
- Verify all service connections
- Test complete lifecycle: prediction → feedback → drift detection → retraining
- Document any issues and fixes

Deliverables
- tests/test_integration.py — integration test suite
- Docker Compose troubleshooting guide

How to Validate
- With all services running in Docker:
- Send prediction request — should succeed
- Send feedback — should be stored
- Wait for drift detector cycle — check logs
- Manually trigger retraining — should complete
- Check MLflow UI accessible at localhost:5001
- Check Prometheus UI at localhost:9090
- Check Grafana UI at localhost:3002
- All dashboards show data
- Run integration test suite — all pass
- Restart services — everything comes back up correctly
