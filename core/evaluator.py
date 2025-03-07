from typing import Dict, Any
import numpy as np
from ..utils.logging import AutoTrainerLogger

class ModelEvaluator:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        self.logger.log("Calculating evaluation metrics")
        
        metrics = {
            'accuracy': self._calculate_accuracy(y_true, y_pred),
            'precision': self._calculate_precision(y_true, y_pred),
            'recall': self._calculate_recall(y_true, y_pred),
            'f1_score': self._calculate_f1_score(y_true, y_pred)
        }
        
        self.logger.log(f"Metrics calculated: {metrics}")
        return metrics
        
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean(y_true == y_pred)
        
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision"""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / (predicted_positives + 1e-7)
        
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall"""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / (actual_positives + 1e-7)
        
    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score"""
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall + 1e-7)