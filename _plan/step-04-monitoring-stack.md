# Step 04: Monitoring Stack (Phase 4)

Time: 4–6 hours

Sub-steps:
- 4.1 Prometheus Setup (1–2 hours)
- 4.2 Grafana Dashboards (2–3 hours)
- 4.3 Custom Metrics (1–2 hours)

---

## 4.1 Prometheus Setup (1–2 hours)

Tasks
- Create prometheus.yml configuration
- Configure scrape targets for prediction and feedback APIs
- Set appropriate scrape intervals
- Define retention policies

Deliverables
- monitoring/prometheus/prometheus.yml

How to Validate
- Start Prometheus with Docker:
  - docker run -d -p 9090:9090 -v $(pwd)/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
- Ensure both BentoML services are running
- Open Prometheus UI at http://localhost:9090
- Check Status > Targets — both services should be "UP"
- Query metrics: try `up` or `prediction_requests_total`
- Verify metrics are being collected (numbers should increase)

---

## 4.2 Grafana Dashboards (2–3 hours)

Tasks
- Set up Grafana with Docker
- Configure Prometheus as datasource
- Create 3 dashboards:
  1. Model Performance (request rate, latency, RMSE)
  2. Drift Monitoring (PSI scores, feature distributions) with annotations/markers for quarter transitions (FD001→FD002→FD003→FD004)
  3. System Health (uptime, errors)
- Export dashboard JSON files
- Set up dashboard provisioning

Deliverables
- monitoring/grafana/dashboards/model_performance.json
- monitoring/grafana/dashboards/drift_monitoring.json
- monitoring/grafana/dashboards/system_health.json
- monitoring/grafana/provisioning/datasources.yml

How to Validate
- Start Grafana: docker run -d -p 3002:3000 grafana/grafana
- Open http://localhost:3002 (login: admin/admin)
- Add Prometheus datasource:
  - URL: http://host.docker.internal:9090 (or Prometheus container name)
  - Test connection — should succeed
- Create dashboard with at least:
  - Prediction request rate panel
  - Prediction latency histogram
  - Current model version
- Verify panels show real data
- Make some prediction requests, watch metrics update in real-time
- Export dashboard JSON and save to repository

---

## 4.3 Custom Metrics (1–2 hours)

Tasks
- Define custom Prometheus metrics in both services
- Add instrumentation for key events
- Create metrics for: predictions, errors, latency, drift scores
- Document all custom metrics

Deliverables
- src/utils/prometheus_metrics.py — metric definitions
- Updated services with metrics instrumentation
- Metrics documentation

How to Validate
- Restart both services with metrics code
- Check /metrics endpoint on both services
- Should see custom metrics like:
  - prediction_requests_total
  - prediction_latency_seconds
  - rul_prediction_value
  - feedback_count_total
- Make predictions and feedback requests
- Verify counter metrics increment
- Check histogram metrics show distribution
- Query custom metrics in Prometheus UI
