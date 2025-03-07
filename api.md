# Full API Reference

## Core Modules

### AutoTrainer
```python
class AutoTrainer:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize AutoTrainer with optional config file"""
    
    def load_data(self, file_path: str) -> None:
        """Load data from file"""
    
    def preprocess_data(self) -> None:
        """Preprocess loaded data"""
    
    def train(self) -> None:
        """Train model using configured settings"""
    
    def evaluate(self) -> dict:
        """Evaluate model performance and return metrics"""
    
    def explain(self, port: int = 8050) -> None:
        """Launch explainability dashboard"""
    
    def save_model(self, path: str) -> None:
        """Save trained model to file"""
    
    def load_model(self, path: str) -> None:
        """Load trained model from file"""

## ConfigManager

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ConfigManager with optional config file"""
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
    
    def save_config(self, config: dict, config_path: str) -> None:
        """Save configuration to file"""
    
    def update_config(self, updates: dict) -> None:
        """Update configuration with new values"""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""

##ModelEvaluator

class ModelEvaluator:
    def __init__(self, model, test_data: Tuple[np.ndarray, np.ndarray]):
        """Initialize with model and test data"""
    
    def calculate_metrics(self) -> dict:
        """Calculate evaluation metrics"""
    
    def confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix"""
    
    def roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve metrics"""


## Utils
## AutoTrainerLogger

class AutoTrainerLogger:
    def __init__(self, name: str = "ai_agent_trainer", level: str = "INFO"):
        """Initialize logger with name and level"""
    
    def log(self, message: str, level: str = 'info') -> None:
        """Log message with specified level"""
    
    def set_level(self, level: str) -> None:
        """Set logging level"""
    
    def get_logs(self) -> List[str]:
        """Get all logged messages"""

## ResourceMonitor

class ResourceMonitor:
    def __init__(self):
        """Initialize resource monitor"""
    
    def get_system_stats(self) -> dict:
        """Get current system resource usage"""
    
    def check_resource_limits(self, limits: dict) -> bool:
        """Check if resource usage exceeds limits"""

## TrainingVisualizer

class TrainingVisualizer:
    def __init__(self):
        """Initialize visualizer"""
    
    def plot_training_history(self, history: dict) -> None:
        """Plot training metrics over time"""
    
    def plot_confusion_matrix(self, y_true, y_pred) -> None:
        """Plot confusion matrix"""
    
    def plot_feature_importance(self, feature_importance: dict) -> None:
        """Plot feature importance"""


## Model Registry
## ModelRegistry

class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry"):
        """Initialize model registry"""
    
    def register_model(self, model, metadata: dict) -> str:
        """Register new model version"""
    
    def load_model(self, version: str):
        """Load specific model version"""
    
    def list_models(self) -> List[dict]:
        """List all registered models"""
    
    def delete_model(self, version: str) -> None:
        """Delete specific model version"""
    
    def get_metadata(self, version: str) -> dict:
        """Get metadata for specific model version"""


## Pipeline
## PipelineOrchestrator

class PipelineOrchestrator:
    def __init__(self):
        """Initialize pipeline orchestrator"""
    
    def register_task(self, name: str, function, dependencies: List[str] = []):
        """Register new pipeline task"""
    
    def run_pipeline(self, config: dict) -> dict:
        """Run complete ML pipeline"""
    
    def schedule_pipeline(self, config: dict, schedule: str) -> None:
        """Schedule pipeline to run periodically"""

## Dashboard
## ExplainabilityDashboard
class ExplainabilityDashboard:
    def __init__(self, model, data):
        """Initialize with model and data"""
    
    def run(self, port: int = 8050) -> None:
        """Run dashboard on specified port"""
    
    def add_custom_tab(self, name: str, content):
        """Add custom tab to dashboard"""


## CI/CD
## CICDManager

class CICDManager:
    def __init__(self):
        """Initialize CI/CD manager"""
    
    def run_tests(self) -> bool:
        """Run all tests in the test suite"""
    
    def run_linting(self) -> bool:
        """Run code linting and style checks"""
    
    def build_docker_image(self, tag: str = "latest") -> bool:
        """Build Docker image for deployment"""
    
    def deploy_to_kubernetes(self, config_path: str) -> bool:
        """Deploy to Kubernetes cluster"""
    
    def run_full_pipeline(self, tag: str = "latest", k8s_config: str = "k8s/deployment.yaml") -> bool:
        """Run full CI/CD pipeline"""