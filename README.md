# Fraud Detection ML System

## Overview
End-to-end ML system for real-time fraud detection using Isolation Forest.

## Project Structure

```
ml/
├── training/                    # Model training pipeline
│   ├── fraud-model/
│   │   ├── train.py            # Main training script
│   │   ├── features.py         # Feature engineering
│   │   ├── dataset.py          # Data loaders
│   │   └── requirements.txt
│   └── notebooks/
│       └── exploratory.ipynb   # EDA and experiments
├── models/                      # Saved model artifacts (gitignored)
│   ├── isolation_forest_v1.joblib
│   └── scaler.pkl
├── inference/
│   ├── fraud-api/              # FastAPI inference service
│   │   ├── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── consumer/               # Kafka consumer service
├── feature-store/
│   └── redis-client.py         # Redis feature fetcher
├── mlflow/
│   └── docker-compose.yml      # Local MLflow server
└── README.md
```

## Quick Start

### 1. Start MLflow Server (Local Development)
```bash
cd mlflow
docker-compose up -d
# Access UI at http://localhost:5000
```

### 2. Train Model
```bash
cd training/fraud-model
pip install -r requirements.txt
python train.py
```

This will:
- Generate synthetic training data (replace with real S3/warehouse data)
- Train Isolation Forest model
- Log to MLflow
- Save model artifacts to `models/`

### 3. Run Inference API
```bash
cd inference/fraud-api
pip install -r requirements.txt

# Set environment variables
export ANOMALY_THRESHOLD=-0.5
export REDIS_HOST=localhost

# Start server
uvicorn main:app --reload --port 8000
```

Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tx_id": "tx_123",
    "user_id": "user_456",
    "amount": 250.00
  }'
```

### 4. Build and Deploy Docker Image
```bash
cd inference/fraud-api
docker build -t fraud-api:v1 .
docker run -p 8000:8000 \
  -e ANOMALY_THRESHOLD=-0.5 \
  -e REDIS_HOST=redis \
  fraud-api:v1
```

## Production Deployment

### Kubernetes
```bash
# Apply manifests (create these based on your infrastructure)
kubectl apply -f k8s/fraud-api-deployment.yaml
kubectl apply -f k8s/fraud-api-service.yaml
kubectl apply -f k8s/fraud-api-ingress.yaml
```

### CI/CD Pipeline
1. **Training**: Trigger via GitHub Actions on new data arrival
2. **Model Registry**: Push to MLflow (backed by S3)
3. **Deployment**: Auto-deploy via ArgoCD when new model version is promoted

## Feature Store Integration

The system uses Redis for online features:
- Transaction velocity (count, amount in time windows)
- User behavior patterns
- Merchant statistics

Features are computed by a separate stream processor (e.g., Flink, Kafka Streams) and stored in Redis.

## Model Details

**Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Why**: Works well for fraud (rare, anomalous patterns) without labels
- **Training Data**: Normal transactions only
- **Features**: Amount, transaction velocity, time-of-day, user history

**Hyperparameters**:
- `n_estimators`: 150
- `contamination`: 0.001 (0.1% expected fraud rate)
- `max_samples`: 256

## Monitoring

- **MLflow**: Track experiments, model versions, metrics
- **Prometheus**: API latency, throughput, error rates
- **CloudWatch/Datadog**: Model predictions, anomaly rate trends
- **A/B Testing**: Compare model versions in production

## TODO
- [ ] Add proper data loaders (S3, Redshift)
- [ ] Implement model retraining schedule
- [ ] Add model performance monitoring
- [ ] Set up A/B testing framework
- [ ] Implement feature drift detection
- [ ] Add batch inference pipeline
- [ ] Create Kubernetes manifests

---

# Reference Repos

1. **Infra:** https://github.com/bshongwe/infra
2. **Fraud Detection ML System:** https://github.com/bshongwe/ml
3. **Services:** https://github.com/bshongwe/services
