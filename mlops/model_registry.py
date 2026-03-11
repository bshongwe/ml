# Model Registry - Track and govern ML models
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import hashlib
import logging

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model approval status."""
    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class ModelRegistry:
    """
    Central registry for ML models.
    Tracks versions, metrics, and deployment status.
    """
    
    def __init__(self, registry_path: str = "s3://ml-registry/models"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
    
    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        training_dataset: str,
        feature_schema: Dict,
        metrics: Dict,
        hyperparameters: Dict,
        framework: str,
        owner: str,
        description: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model (e.g., 'fraud_detector')
            model_version: Version string (e.g., 'v1.2.0')
            model_path: S3/local path to model artifacts
            training_dataset: Dataset fingerprint/path
            feature_schema: Feature definitions and types
            metrics: Performance metrics (accuracy, precision, etc.)
            hyperparameters: Model hyperparameters
            framework: ML framework (sklearn, pytorch, etc.)
            owner: Team/person responsible
            description: Model description
            tags: Optional tags for categorization
        
        Returns:
            Model ID (unique identifier)
        """
        # Generate unique model ID
        model_id = self._generate_model_id(model_name, model_version)
        
        # Create model metadata
        metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'model_version': model_version,
            'model_path': model_path,
            'training_dataset': training_dataset,
            'dataset_fingerprint': self._hash_string(training_dataset),
            'feature_schema': feature_schema,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'framework': framework,
            'owner': owner,
            'description': description,
            'tags': tags or [],
            'status': ModelStatus.EXPERIMENTAL.value,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'approved_by': None,
            'approval_date': None,
            'deployment_history': []
        }
        
        # Save to registry
        self._save_model_metadata(model_id, metadata)
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def update_model_status(
        self,
        model_id: str,
        status: ModelStatus,
        approved_by: Optional[str] = None
    ):
        """
        Update model approval status.
        
        Args:
            model_id: Unique model identifier
            status: New status
            approved_by: Approver name (required for production)
        """
        metadata = self.get_model_metadata(model_id)
        
        # Validate production promotion
        if status == ModelStatus.PRODUCTION:
            if not approved_by:
                raise ValueError("Production deployment requires approver")
            
            # Check if metrics meet thresholds
            if not self._validate_production_metrics(metadata['metrics']):
                raise ValueError("Model metrics do not meet production thresholds")
        
        # Update metadata
        metadata['status'] = status.value
        metadata['updated_at'] = datetime.now().isoformat()
        
        if approved_by:
            metadata['approved_by'] = approved_by
            metadata['approval_date'] = datetime.now().isoformat()
        
        self._save_model_metadata(model_id, metadata)
        
        logger.info(f"Updated model {model_id} status to {status.value}")
    
    def get_model_metadata(self, model_id: str) -> Dict:
        """Get full model metadata."""
        metadata_path = self.registry_path / f"{model_id}.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Model {model_id} not found in registry")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """
        Get the current production model for a given name.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model metadata or None if no production model exists
        """
        models = self.list_models(
            model_name=model_name,
            status=ModelStatus.PRODUCTION
        )
        
        if not models:
            return None
        
        # Return most recent production model
        return sorted(models, key=lambda x: x['created_at'], reverse=True)[0]
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        List models with optional filters.
        
        Args:
            model_name: Filter by model name
            status: Filter by status
            tags: Filter by tags
        
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        for metadata_file in self.registry_path.glob("*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Apply filters
            if model_name and metadata['model_name'] != model_name:
                continue
            
            if status and metadata['status'] != status.value:
                continue
            
            if tags and not any(tag in metadata['tags'] for tag in tags):
                continue
            
            models.append(metadata)
        
        return models
    
    def add_deployment_event(
        self,
        model_id: str,
        environment: str,
        deployed_by: str,
        deployment_config: Dict
    ):
        """
        Record a deployment event.
        
        Args:
            model_id: Model identifier
            environment: Deployment environment (staging/production)
            deployed_by: Person/system that deployed
            deployment_config: Deployment configuration
        """
        metadata = self.get_model_metadata(model_id)
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'environment': environment,
            'deployed_by': deployed_by,
            'config': deployment_config
        }
        
        metadata['deployment_history'].append(event)
        metadata['updated_at'] = datetime.now().isoformat()
        
        self._save_model_metadata(model_id, metadata)
        
        logger.info(f"Recorded deployment of {model_id} to {environment}")
    
    def compare_models(self, model_id_1: str, model_id_2: str) -> Dict:
        """
        Compare metrics between two models.
        
        Returns:
            Dictionary with metric comparisons
        """
        model_1 = self.get_model_metadata(model_id_1)
        model_2 = self.get_model_metadata(model_id_2)
        
        comparison = {
            'model_1': {
                'id': model_id_1,
                'version': model_1['model_version'],
                'metrics': model_1['metrics']
            },
            'model_2': {
                'id': model_id_2,
                'version': model_2['model_version'],
                'metrics': model_2['metrics']
            },
            'differences': {}
        }
        
        # Compare metrics
        for metric in model_1['metrics']:
            if metric in model_2['metrics']:
                diff = model_2['metrics'][metric] - model_1['metrics'][metric]
                comparison['differences'][metric] = {
                    'absolute_diff': diff,
                    'relative_diff_pct': (diff / model_1['metrics'][metric]) * 100
                }
        
        return comparison
    
    def _generate_model_id(self, model_name: str, model_version: str) -> str:
        """Generate unique model ID."""
        return f"{model_name}_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _hash_string(self, s: str) -> str:
        """Generate hash fingerprint."""
        return hashlib.sha256(s.encode()).hexdigest()[:16]
    
    def _save_model_metadata(self, model_id: str, metadata: Dict):
        """Save model metadata to registry."""
        metadata_path = self.registry_path / f"{model_id}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _validate_production_metrics(self, metrics: Dict) -> bool:
        """
        Validate that metrics meet production thresholds.
        Override this with your specific requirements.
        """
        # Example thresholds
        thresholds = {
            'precision': 0.95,
            'recall': 0.90,
            'f1_score': 0.92
        }
        
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                logger.warning(f"Metric {metric} ({metrics[metric]}) below threshold ({threshold})")
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    registry = ModelRegistry(registry_path="./model_registry")
    
    # Register a model
    model_id = registry.register_model(
        model_name="fraud_detector",
        model_version="v1.0.0",
        model_path="s3://models/fraud_v1.joblib",
        training_dataset="s3://data/transactions_2024_q1.parquet",
        feature_schema={
            'amount': 'float',
            'tx_count_24h': 'int',
            'avg_amount_7d': 'float'
        },
        metrics={
            'precision': 0.96,
            'recall': 0.93,
            'f1_score': 0.945,
            'auc_roc': 0.98
        },
        hyperparameters={
            'n_estimators': 150,
            'contamination': 0.001,
            'max_samples': 256
        },
        framework="scikit-learn",
        owner="fraud-team",
        description="Isolation Forest for fraud detection",
        tags=["fraud", "unsupervised", "production-ready"]
    )
    
    print(f"Registered model: {model_id}")
    
    # Promote to production
    registry.update_model_status(
        model_id=model_id,
        status=ModelStatus.PRODUCTION,
        approved_by="ml-platform-lead"
    )
    
    # Record deployment
    registry.add_deployment_event(
        model_id=model_id,
        environment="production",
        deployed_by="cicd-pipeline",
        deployment_config={
            'replicas': 3,
            'memory': '2Gi',
            'cpu': '1000m'
        }
    )
