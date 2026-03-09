# train.py - Run locally or in CI/CD/SageMaker
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'  # or SageMaker/ local

mlflow.set_experiment("fraud-detection")

with mlflow.start_run():
    # Stub: Load normal transactions (replace with real S3/warehouse read)
    # df = pd.read_parquet("s3://bucket/normal_tx.parquet")
    # For demo: synthetic data
    np.random.seed(42)
    normal_data = np.random.normal(loc=100, scale=50, size=(10000, 2))  # e.g., amount, time-of-day
    anomalies = np.random.normal(loc=1000, scale=200, size=(100, 2))   # fake fraud
    X = np.vstack([normal_data, anomalies])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        n_estimators=150,
        contamination=0.001,  # expected fraud rate ~0.1%
        random_state=42,
        max_samples=256
    )
    
    model.fit(X_scaled)
    
    # Log params & model
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("contamination", model.contamination)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("scaler.pkl")  # save scaler too
    
    # Save locally for consumer
    joblib.dump(model, "models/isolation_forest_v1.joblib")
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("Model trained and logged to MLflow")
