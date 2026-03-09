import os
import redis
import json

r = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis.data.svc.cluster.local'),
    port=6379,
    decode_responses=True
)

def get_user_features(user_id: str) -> dict:
    """Fetch online features (e.g., tx velocity, last tx time)"""
    key = f"user_features:{user_id}"
    raw = r.get(key)
    if raw:
        return json.loads(raw)
    return {"tx_count_24h": 0, "avg_amount_24h": 0.0}  # fallback

def update_user_features(user_id: str, amount: float):
    """Update after transaction (called by Transaction service or stream processor)"""
    key = f"user_features:{user_id}"
    # Increment count, recalculate avg... (use Redis pipelines for atomicity)
    pipe = r.pipeline()
    pipe.incr(f"{key}:count")
    # ... more logic
    pipe.execute()
