# Step 09 (Optional): Kubernetes Deployment (Phase 9)

Time: 6–10 hours

Sub-steps:
- 9.1 Kubernetes Manifests (3–4 hours)
- 9.2 Minikube Setup (2–3 hours)
- 9.3 K8s Testing (1–2 hours)
- 9.4 K8s Documentation (1 hour)

---

## 9.1 Kubernetes Manifests (3–4 hours)

Tasks
- Create Deployment manifests for all services
- Create Service manifests for networking
- Create ConfigMaps for configurations
- Create PersistentVolumeClaims for data
- Create Ingress (optional)

Deliverables
- k8s/deployment-prediction.yaml
- k8s/deployment-feedback.yaml
- k8s/deployment-retraining.yaml
- k8s/deployment-drift-detector.yaml
- k8s/deployment-mlflow.yaml
- k8s/deployment-prometheus.yaml
- k8s/deployment-grafana.yaml
- k8s/service-*.yaml for each deployment
- k8s/configmap.yaml
- k8s/pvc.yaml

---

## 9.2 Minikube Setup (2–3 hours)

Tasks
- Install minikube
- Configure resource allocation
- Deploy all services to local K8s cluster
- Set up port forwarding for access
- Test service discovery

Deliverables
- Minikube configuration
- Deployment scripts
- Port forwarding commands documented

How to Validate
- Minikube running: minikube status
- All pods running: kubectl get pods --all-namespaces
- Set up port forwards:
  - kubectl port-forward svc/prediction-api 3000:3000
  - kubectl port-forward svc/grafana 3002:3000
- Test prediction API through port forward
- Check Grafana accessible
- Verify inter-pod communication:
  - Prediction API can reach MLflow service
  - Drift detector can reach feedback API
- Scale prediction API: kubectl scale deployment prediction-api --replicas=3
- Verify load balancing works

---

## 9.3 K8s Testing (1–2 hours)

Tasks
- Test all deployments
- Test service discovery
- Test rolling updates
- Test pod restart resilience
- Test scaling

Deliverables
- K8s test suite
- Scaling test results

How to Validate
- Perform rolling update:
  - Update image version in deployment
  - Apply: kubectl apply -f k8s/deployment-prediction.yaml
  - Watch: kubectl rollout status deployment/prediction-api
  - Verify zero downtime (keep making requests during update)
- Test pod failure:
  - Delete a pod: kubectl delete pod <pod-name>
  - Verify K8s recreates it automatically
  - Service should continue working
- Test scaling:
  - Scale up: kubectl scale deployment prediction-api --replicas=5
  - Verify 5 pods running
  - Test load distribution
- Check resource usage: kubectl top pods
- All services remain healthy during tests

---

## 9.4 K8s Documentation (1 hour)

Tasks
- Document K8s deployment process
- Add kubectl command reference
- Create troubleshooting guide for K8s issues
- Document architecture differences from Docker Compose

Deliverables
- K8s deployment guide
- kubectl cheat sheet
- K8s troubleshooting section

How to Validate
- Follow documentation to deploy from scratch
- All kubectl commands work
- Troubleshooting guide covers common issues:
  - ImagePullBackOff
  - CrashLoopBackOff
  - Service not accessible
  - PVC not binding
- Documentation explains when to use K8s vs Docker Compose
- Architecture diagram updated to show K8s components
