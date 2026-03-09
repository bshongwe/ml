# Data loader stub
import pandas as pd
import boto3
from typing import Optional
import os

def load_training_data(
    source: str = "s3",
    s3_bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
    local_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load training data from various sources.
    
    Args:
        source: 's3', 'local', or 'redshift'
        s3_bucket: S3 bucket name
        s3_key: S3 object key (e.g., 'data/transactions/2024/*.parquet')
        local_path: Local file path
    
    Returns:
        DataFrame with transaction data
    """
    if source == "s3":
        if not s3_bucket or not s3_key:
            raise ValueError("s3_bucket and s3_key required for S3 source")
        
        s3 = boto3.client('s3')
        # Download or read directly with pandas
        # df = pd.read_parquet(f"s3://{s3_bucket}/{s3_key}")
        
        # For demo purposes:
        print(f"Would load from s3://{s3_bucket}/{s3_key}")
        return pd.DataFrame()
    
    elif source == "local":
        if not local_path:
            raise ValueError("local_path required for local source")
        return pd.read_parquet(local_path)
    
    elif source == "redshift":
        # Use redshift_connector or psycopg2
        # query = "SELECT * FROM transactions WHERE is_fraud = 0 AND date >= '2024-01-01'"
        # df = pd.read_sql(query, connection)
        raise NotImplementedError("Redshift loader not implemented")
    
    else:
        raise ValueError(f"Unknown source: {source}")

def filter_normal_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to normal transactions for unsupervised training.
    """
    # Remove known fraud
    if 'is_fraud' in df.columns:
        df = df[df['is_fraud'] == 0]
    
    # Remove outliers (optional - be careful not to remove legitimate edge cases)
    # df = df[df['amount'] < df['amount'].quantile(0.999)]
    
    return df
