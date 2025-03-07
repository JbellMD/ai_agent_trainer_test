import numpy as np
from ..utils.logging import AutoTrainerLogger

class ModelMonitor:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def detect_data_drift(self, reference_data, current_data, threshold: float = 0.1):
        """Detect data drift using statistical tests"""
        self.logger.log("Checking for data drift")
        # Example implementation using Kolmogorov-Smirnov test
        from scipy.stats import ks_2samp
        p_values = []
        for col in reference_data.columns:
            stat, p_value = ks_2samp(reference_data[col], current_data[col])
            p_values.append(p_value)
        return np.mean(p_values) < threshold
        
    def check_model_accuracy(self, y_true, y_pred, threshold: float = 0.8):
        """Check if model accuracy is above threshold"""
        self.logger.log("Checking model accuracy")
        accuracy = (y_true == y_pred).mean()
        return accuracy >= threshold