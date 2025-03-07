"""AI Agent Trainer - Automated Machine Learning System

This package provides a comprehensive framework for autonomous machine learning
model creation and optimization. It includes modules for data processing,
model selection, training, evaluation, and deployment.

Main Components:
- Core: AutoTrainer, ConfigManager, ModelEvaluator
- Modules: Data processing, model selection, optimization
- Utils: Logging, monitoring, visualization
- Deployment: Model registry, CI/CD pipelines
- Dashboard: Model explainability and monitoring

"""

from .core import AutoTrainer, ConfigManager, ModelEvaluator
from .utils import AutoTrainerLogger, ResourceMonitor, TrainingVisualizer
from .model_registry import ModelRegistry
from .pipeline import PipelineOrchestrator
from .dashboard import ExplainabilityDashboard

__version__ = "1.0.0"
__all__ = [
    'AutoTrainer',
    'ConfigManager',
    'ModelEvaluator',
    'AutoTrainerLogger',
    'ResourceMonitor',
    'TrainingVisualizer',
    'ModelRegistry',
    'PipelineOrchestrator',
    'ExplainabilityDashboard'
]

# Initialize logging configuration
from .utils.logging import setup_logging
setup_logging()