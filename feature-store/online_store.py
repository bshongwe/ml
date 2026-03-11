# Online Feature Store - Low-latency feature serving
import redis
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """
    Online feature store for low-latency inference (<10ms).
    Uses Redis for sub-millisecond feature retrieval.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        self.default_ttl = 86400  # 24 hours
    
    def write_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict,
        ttl: Optional[int] = None
    ):
        """
        Write features to online store.
        
        Args:
            entity_type: Type of entity (e.g., 'user', 'merchant')
            entity_id: Unique entity identifier
            features: Feature key-value pairs
            ttl: Time to live in seconds (default: 24h)
        """
        key = f"{entity_type}:{entity_id}:features"
        
        # Add metadata
        features['_updated_at'] = datetime.now().isoformat()
        features['_entity_type'] = entity_type
        features['_entity_id'] = entity_id
        
        # Write to Redis
        self.redis_client.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(features)
        )
        
        logger.debug(f"Wrote features for {entity_type}:{entity_id}")
    
    def read_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Read features from online store.
        
        Args:
            entity_type: Type of entity
            entity_id: Unique entity identifier
            feature_names: Specific features to retrieve (None = all)
        
        Returns:
            Dictionary of feature values
        """
        key = f"{entity_type}:{entity_id}:features"
        
        try:
            raw = self.redis_client.get(key)
            if not raw:
                logger.warning(f"No features found for {entity_type}:{entity_id}")
                return {}
            
            features = json.loads(raw)
            
            # Filter to requested features
            if feature_names:
                features = {k: v for k, v in features.items() if k in feature_names}
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to read features: {e}")
            return {}
    
    def batch_read_features(
        self,
        entity_type: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Batch read features for multiple entities.
        More efficient than individual reads.
        
        Args:
            entity_type: Type of entity
            entity_ids: List of entity identifiers
            feature_names: Specific features to retrieve
        
        Returns:
            Dictionary mapping entity_id -> features
        """
        keys = [f"{entity_type}:{eid}:features" for eid in entity_ids]
        
        # Use Redis pipeline for batch reads
        pipe = self.redis_client.pipeline()
        for key in keys:
            pipe.get(key)
        
        results = pipe.execute()
        
        # Parse results
        features_by_entity = {}
        for entity_id, raw in zip(entity_ids, results):
            if raw:
                features = json.loads(raw)
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                features_by_entity[entity_id] = features
            else:
                features_by_entity[entity_id] = {}
        
        return features_by_entity
    
    def update_feature(
        self,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        value: any
    ):
        """
        Update a single feature without overwriting others.
        """
        features = self.read_features(entity_type, entity_id)
        features[feature_name] = value
        self.write_features(entity_type, entity_id, features)
    
    def increment_counter(
        self,
        entity_type: str,
        entity_id: str,
        counter_name: str,
        amount: int = 1
    ) -> int:
        """
        Atomically increment a counter feature.
        Useful for transaction counts, etc.
        
        Returns:
            New counter value
        """
        key = f"{entity_type}:{entity_id}:counter:{counter_name}"
        new_value = self.redis_client.incr(key, amount)
        self.redis_client.expire(key, self.default_ttl)
        return new_value
    
    def get_feature_freshness(
        self,
        entity_type: str,
        entity_id: str
    ) -> Optional[timedelta]:
        """
        Check how old the features are.
        Important for detecting stale data.
        
        Returns:
            Age of features or None if not found
        """
        features = self.read_features(entity_type, entity_id)
        
        if '_updated_at' in features:
            updated_at = datetime.fromisoformat(features['_updated_at'])
            return datetime.now() - updated_at
        
        return None


# Example usage
if __name__ == "__main__":
    online_store = OnlineFeatureStore()
    
    # Write features
    online_store.write_features(
        entity_type='user',
        entity_id='user_123',
        features={
            'tx_count_24h': 15,
            'avg_amount_7d': 234.56,
            'unique_merchants_24h': 7,
            'risk_score': 0.23
        }
    )
    
    # Read features
    features = online_store.read_features('user', 'user_123')
    print(f"Features: {features}")
    
    # Increment counter
    new_count = online_store.increment_counter('user', 'user_123', 'tx_count')
    print(f"New transaction count: {new_count}")
    
    # Check freshness
    age = online_store.get_feature_freshness('user', 'user_123')
    print(f"Feature age: {age}")
