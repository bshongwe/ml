from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add feature-store to path for imports
sys.path.append('/app/feature-store')
from redis_client import get_user_features

app = FastAPI(title="Fraud Inference API")

model = joblib.load("/app/models/isolation_forest_v1.joblib")
scaler = joblib.load("/app/models/scaler.pkl")

class TransactionEvent(BaseModel):
    tx_id: str
    user_id: str
    amount: float
    # add more fields: merchant, time, location_delta, etc.

@app.post("/predict")
async def predict(event: TransactionEvent):
    try:
        # Fetch features
        features_dict = get_user_features(event.user_id)  # from Redis
        features = np.array([[event.amount, features_dict.get('tx_count_24h', 0)]])  # expand!
        scaled = scaler.transform(features)
        
        score = model.score_samples(scaled)[0]
        is_anomaly = score < float(os.getenv("ANOMALY_THRESHOLD", "-0.5"))
        
        return {
            "tx_id": event.tx_id,
            "anomaly_score": float(score),
            "is_fraud": is_anomaly,
            "reason": "High anomaly" if is_anomaly else "Normal"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
