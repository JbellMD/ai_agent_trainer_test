from torch.optim import Adam, SGD
from ..utils.logging import AutoTrainerLogger

class ModelOptimizer:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def get_optimizer(self, model, optimizer_name: str, learning_rate: float):
        """Get optimizer by name"""
        self.logger.log(f"Creating {optimizer_name} optimizer with lr={learning_rate}")
        optimizers = {
            'adam': Adam(model.parameters(), lr=learning_rate),
            'sgd': SGD(model.parameters(), lr=learning_rate)
        }
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        return optimizers[optimizer_name]