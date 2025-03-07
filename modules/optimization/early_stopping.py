import numpy as np
from ..utils.logging import AutoTrainerLogger

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.logger = AutoTrainerLogger()
        
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.logger.log("Early stopping triggered")
                return True
        return False