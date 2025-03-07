import numpy as np
from ..utils.logging import AutoTrainerLogger
from ..modules.model_selection import ModelSelector, HyperparameterTuner

class AutoMLEngine:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        self.model_selector = ModelSelector()
        self.tuner = HyperparameterTuner()
        
    def run_automl(self, X, y, time_limit: int = 3600):
        """Run automated machine learning process"""
        self.logger.log(f"Starting AutoML with time limit: {time_limit} seconds")
        best_model = None
        best_score = -np.inf
        
        for model_name in self.model_selector.get_available_models():
            model = self.model_selector.select_model(model_name)
            tuned_model = self.tuner.tune_hyperparameters(
                model,
                param_grid=self._get_default_param_grid(model_name),
                X=X,
                y=y
            )
            score = tuned_model.score(X, y)
            if score > best_score:
                best_score = score
                best_model = tuned_model
                
        self.logger.log(f"AutoML completed. Best model: {best_model}")
        return best_model
        
    def _get_default_param_grid(self, model_name: str):
        """Get default parameter grid for each model type"""
        if model_name == 'random_forest':
            return {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20]
            }
        elif model_name == 'logistic_regression':
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        # Add more model-specific parameter grids