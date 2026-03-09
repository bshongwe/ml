# Feature engineering
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def extract_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from raw transaction data.
    
    Args:
        df: DataFrame with columns [tx_id, user_id, amount, timestamp, merchant_id, etc.]
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Time-based features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(0, 6).astype(int)
    
    # Amount-based features
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_rounded'] = (df['amount'] % 1 == 0).astype(int)  # Round number indicator
    
    # User aggregations (rolling windows - compute in stream/batch)
    # These would typically come from feature store
    # df['user_tx_count_24h'] = ...
    # df['user_avg_amount_7d'] = ...
    # df['user_std_amount_7d'] = ...
    
    return df

def compute_velocity_features(user_id: str, current_time: datetime, lookback_hours: int = 24) -> dict:
    """
    Compute velocity features for a user.
    In production, query from time-series DB or feature store.
    
    Returns:
        dict with velocity metrics
    """
    # Stub - replace with actual query
    return {
        'tx_count_1h': 0,
        'tx_count_24h': 0,
        'total_amount_1h': 0.0,
        'total_amount_24h': 0.0,
        'unique_merchants_24h': 0,
        'avg_amount_7d': 100.0,
        'std_amount_7d': 50.0
    }
