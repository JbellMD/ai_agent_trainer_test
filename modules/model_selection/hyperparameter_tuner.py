from sklearn.model_selection import GridSearchCV
from ..utils.logging import AutoTrainerLogger

class HyperparameterTuner:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def tune_hyperparameters(self, model, param_grid, X, y):
        """Tune hyperparameters using grid search"""
        self.logger.log("Starting hyperparameter tuning")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy'
        )
        grid_search.fit(X, y)
        self.logger.log(f"Best parameters found: {grid_search.best_params_}")
        return grid_search.best_estimator_