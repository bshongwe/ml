# Offline Feature Store - Historical features for training
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OfflineFeatureStore:
    """
    Offline feature store for historical training data.
    Stores features in Parquet format in the data lake.
    """
    
    def __init__(self, base_path: str = "s3://feature-store/offline"):
        self.base_path = Path(base_path)
        
    def write_features(
        self,
        feature_group: str,
        df: pd.DataFrame,
        partition_cols: Optional[List[str]] = None
    ):
        """
        Write features to offline store.
        
        Args:
            feature_group: Name of feature group (e.g., 'fraud_features')
            df: DataFrame with features
            partition_cols: Columns to partition by (e.g., ['year', 'month'])
        """
        output_path = self.base_path / feature_group
        
        if partition_cols:
            df.to_parquet(
                output_path,
                engine='pyarrow',
                partition_cols=partition_cols,
                compression='snappy'
            )
        else:
            df.to_parquet(
                output_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                engine='pyarrow',
                compression='snappy'
            )
        
        logger.info(f"Wrote {len(df)} records to {feature_group}")
    
    def read_features(
        self,
        feature_group: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filters: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Read features from offline store.
        
        Args:
            feature_group: Name of feature group
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            filters: PyArrow filters for partitioned data
        
        Returns:
            DataFrame with features
        """
        feature_path = self.base_path / feature_group
        
        # Build filters for date range
        if start_date or end_date:
            filters = filters or []
            if start_date:
                filters.append(('date', '>=', start_date))
            if end_date:
                filters.append(('date', '<=', end_date))
        
        try:
            df = pd.read_parquet(feature_path, filters=filters)
            logger.info(f"Read {len(df)} records from {feature_group}")
            return df
        except Exception as e:
            logger.error(f"Failed to read features: {e}")
            raise
    
    def get_point_in_time_features(
        self,
        feature_group: str,
        entity_ids: List[str],
        timestamp: datetime
    ) -> pd.DataFrame:
        """
        Get features as they existed at a specific point in time.
        Critical for training to avoid data leakage.
        
        Args:
            feature_group: Name of feature group
            entity_ids: List of entity IDs (e.g., user_ids)
            timestamp: Point in time to retrieve features
        
        Returns:
            DataFrame with historical features
        """
        df = self.read_features(
            feature_group,
            end_date=timestamp.strftime('%Y-%m-%d')
        )
        
        # Get latest features before timestamp for each entity
        df_filtered = df[
            (df['entity_id'].isin(entity_ids)) &
            (df['timestamp'] <= timestamp)
        ].sort_values('timestamp').groupby('entity_id').last()
        
        return df_filtered


class FeatureRegistry:
    """
    Registry for feature definitions and metadata.
    Ensures consistency between training and serving.
    """
    
    def __init__(self):
        self.features: Dict[str, Dict] = {}
    
    def register_feature_group(
        self,
        name: str,
        features: List[str],
        entity: str,
        description: str,
        owner: str,
        schema: Dict
    ):
        """
        Register a feature group.
        
        Args:
            name: Feature group name
            features: List of feature names
            entity: Entity type (e.g., 'user', 'transaction')
            description: Human-readable description
            owner: Team/person responsible
            schema: Feature schema (types, constraints)
        """
        self.features[name] = {
            'features': features,
            'entity': entity,
            'description': description,
            'owner': owner,
            'schema': schema,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        logger.info(f"Registered feature group: {name}")
    
    def get_feature_group(self, name: str) -> Dict:
        """Get feature group metadata."""
        if name not in self.features:
            raise ValueError(f"Feature group {name} not found")
        return self.features[name]
    
    def list_feature_groups(self) -> List[str]:
        """List all registered feature groups."""
        return list(self.features.keys())


# Example usage
if __name__ == "__main__":
    # Initialize stores
    offline_store = OfflineFeatureStore()
    registry = FeatureRegistry()
    
    # Register feature group
    registry.register_feature_group(
        name='fraud_features',
        features=['tx_count_24h', 'avg_amount_7d', 'unique_merchants_24h'],
        entity='user',
        description='User transaction velocity features',
        owner='fraud-detection-team',
        schema={
            'tx_count_24h': 'int',
            'avg_amount_7d': 'float',
            'unique_merchants_24h': 'int'
        }
    )
    
    # Write features
    sample_df = pd.DataFrame({
        'entity_id': ['user_1', 'user_2'],
        'tx_count_24h': [5, 12],
        'avg_amount_7d': [100.5, 250.3],
        'unique_merchants_24h': [3, 8],
        'timestamp': [datetime.now(), datetime.now()]
    })
    
    offline_store.write_features('fraud_features', sample_df)
