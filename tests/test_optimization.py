import pytest
from ai_agent_trainer.modules.optimization import EarlyStopping

class TestOptimization:
    def test_early_stopping(self):
        stopper = EarlyStopping(patience=2)
        # Simulate validation loss
        losses = [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]
        should_stop = [stopper(loss) for loss in losses]
        assert should_stop[-1] == True