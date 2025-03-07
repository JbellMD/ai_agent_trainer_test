import shap
import numpy as np
from ..utils.logging import AutoTrainerLogger

class ModelExplainability:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def explain_model(self, model, X):
        """Generate model explanations using SHAP"""
        self.logger.log("Generating model explanations")
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        return shap_values
        
    def feature_importance(self, model, X):
        """Calculate feature importance"""
        self.logger.log("Calculating feature importance")
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            # For models without built-in feature importance
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            return np.abs(shap_values.values).mean(0)