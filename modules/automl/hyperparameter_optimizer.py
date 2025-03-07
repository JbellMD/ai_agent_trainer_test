from optuna import create_study
from optuna.samplers import TPESampler
from ..utils.logging import AutoTrainerLogger

class HyperparameterOptimizer:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def optimize_hyperparameters(self, model, X, y, n_trials: int = 100):
        """Optimize hyperparameters using Bayesian optimization"""
        self.logger.log(f"Starting hyperparameter optimization with {n_trials} trials")
        study = create_study(direction='maximize', sampler=TPESampler())
        study.optimize(lambda trial: self._objective(trial, model, X, y), n_trials=n_trials)
        return study.best_params
        
    def _objective(self, trial, model, X, y):
        """Objective function for optimization"""
        params = self._get_hyperparameter_space(trial, model)
        model.set_params(**params)
        from sklearn.model_selection import cross_val_score
        return cross_val_score(model, X, y, cv=5).mean()
        
    def _get_hyperparameter_space(self, trial, model):
        """Define hyperparameter search space"""
        if isinstance(model, RandomForestClassifier):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
            }
        # Add more model-specific parameter spaces