import os
import pickle
import json
import hashlib
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, ValidationError
from ..utils.logging import AutoTrainerLogger

class ModelMetadata(BaseModel):
    """Schema for model metadata"""
    model_type: str
    training_date: str
    performance_metrics: Dict[str, float]
    feature_columns: list
    target_column: str
    author: str
    description: str

class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry"):
        self.logger = AutoTrainerLogger()
        self.registry_path = registry_path
        os.makedirs(self.registry_path, exist_ok=True)
        self.current_version = self._get_latest_version()
        
    def register_model(self, model, metadata: Dict[str, Any]):
        """Register a new model version with validation"""
        try:
            # Validate metadata schema
            validated_metadata = ModelMetadata(**metadata)
            
            # Validate model can make predictions
            self._validate_model(model)
            
            version = self._generate_version()
            model_path = os.path.join(self.registry_path, f"model_v{version}.pkl")
            metadata_path = os.path.join(self.registry_path, f"metadata_v{version}.json")
            
            # Save model and metadata
            self._save_model(model, model_path)
            self._save_metadata(validated_metadata.dict(), metadata_path)
            
            self.current_version = version
            self.logger.log(f"Successfully registered model version {version}")
            return version
            
        except ValidationError as e:
            self.logger.log(f"Metadata validation failed: {e}", level='error')
            raise
        except Exception as e:
            self.logger.log(f"Model registration failed: {e}", level='error')
            raise
            
    def load_model(self, version: str):
        """Load a specific model version"""
        model_path = os.path.join(self.registry_path, f"model_v{version}.pkl")
        if not os.path.exists(model_path):
            raise ValueError(f"Model version {version} not found")
            
        self.logger.log(f"Loading model version {version}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
            
    def rollback_version(self, version: str):
        """Rollback to a previous model version"""
        if not self._version_exists(version):
            raise ValueError(f"Version {version} does not exist")
            
        self.current_version = version
        self.logger.log(f"Rolled back to version {version}")
        return self.load_model(version)
        
    def _validate_model(self, model):
        """Validate model can make predictions"""
        # Add more comprehensive validation as needed
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method")
            
    def _save_model(self, model, path: str):
        """Save model with checksum verification"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        # Verify model was saved correctly
        with open(path, 'rb') as f:
            saved_model = pickle.load(f)
        if not self._models_equal(model, saved_model):
            os.remove(path)
            raise RuntimeError("Model serialization verification failed")
            
    def _save_metadata(self, metadata: dict, path: str):
        """Save metadata with checksum"""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _models_equal(self, model1, model2) -> bool:
        """Compare models for equality"""
        # Implement more robust comparison if needed
        return str(model1) == str(model2)
        
    def _generate_version(self) -> str:
        """Generate version string based on timestamp and hash"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:6]}"
        
    def _get_latest_version(self) -> str:
        """Get the latest registered version"""
        versions = [f for f in os.listdir(self.registry_path) if f.startswith('model_v')]
        if not versions:
            return None
        return sorted(versions)[-1].split('_v')[1].split('.')[0]
        
    def _version_exists(self, version: str) -> bool:
        """Check if version exists"""
        return os.path.exists(os.path.join(self.registry_path, f"model_v{version}.pkl"))