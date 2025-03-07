import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
from .logging import AutoTrainerLogger

class TrainingVisualizer:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def plot_training_history(self, history: Dict[str, Any]):
        """Plot training metrics over time"""
        self.logger.log("Generating training history plot")
        df = pd.DataFrame(history)
        
        plt.figure(figsize=(12, 6))
        for metric in df.columns:
            plt.plot(df[metric], label=metric)
            
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        self.logger.log("Generating confusion matrix")
        # Implement confusion matrix visualization
        pass
        
    def plot_feature_importance(self, feature_importance: Dict[str, float]):
        """Plot feature importance"""
        self.logger.log("Generating feature importance plot")
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance.keys(), feature_importance.values())
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.show()