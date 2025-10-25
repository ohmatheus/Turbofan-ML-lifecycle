# Step 07: Production Simulation & Demo (Phase 7)

Time: 3–4 hours

Sub-steps:
- 7.1 Production Simulator (2 hours)
- 7.2 End-to-End Demo (1–2 hours)

---

## 7.1 Production Simulator (2 hours)

Tasks
- Create script to stream engine data to prediction API
- Implement quarter-based simulation using C-MAPSS subsets: FD001 → FD002 → FD003 → FD004
- At each quarter boundary, sporadically inject a small batch of engine units from the upcoming subset to simulate progressive drift
- Add configurable request rate and sampling to keep runtime ≈ 10 minutes
- Send both predictions and feedback (with delay)

Deliverables
- simulation/production_simulator.py — main simulator
- simulation/drift_scenarios.py — scenario configurations

How to Validate
- Run simulator: python simulation/production_simulator.py --subsets FD001,FD002,FD003,FD004 --rate 10/min --mix-at-boundaries
- Should send prediction requests at specified rate
- Check prediction service logs show incoming requests
- Check feedback service receives feedback data
- Run for 5–10 minutes
- Verify Grafana shows:
  - Increasing request counts
  - Prediction distribution
  - Drift metrics gradually rising as stream transitions FD001→FD002→FD003→FD004 with small mixed batches at boundaries
  - Feedback accumulating
- Stop and restart — should continue without issues
- Test different scenarios (gradual, sudden drift)

---

## 7.2 End-to-End Demo (1–2 hours)

Tasks
- Create demonstration script
- Run complete workflow from clean state
- Document each step with expected outcomes
- Capture screenshots of key moments
- Prepare talking points

Deliverables
- Demo script/runbook
- Screenshots of Grafana showing drift
- Video recording (optional)
- Presentation talking points

How to Validate
- Start from scratch: docker-compose down -v && docker-compose up
- Run demo script:
  1. Show initial state — Model v1.0 serving
  2. Start production simulator with Subsets FD001→FD004 (with boundary mixing)
  3. Watch Grafana dashboard
  4. Observe drift metrics climbing
  5. Wait for automatic retraining trigger
  6. Show new model v1.1 deployed
  7. Observe metrics improve
- Capture screenshots at each step
- Verify entire cycle completes in < 30 minutes
- Document timing for each phase
- Prepare 5-minute verbal walkthrough
