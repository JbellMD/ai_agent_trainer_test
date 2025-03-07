import pytest
import numpy as np
from ai_agent_trainer.modules.model_selection import ModelSelector, HyperparameterTuner

class TestModelSelection:
    @pytest.fixture
    def sample_data(self):
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_model_selector(self):
        selector = ModelSelector()
        models = selector.get_available_models()
        assert len(models) > 0
        for model_name in models:
            model = selector.select_model(model_name)
            assert model is not None

    def test_hyperparameter_tuning(self, sample_data):
        X, y = sample_data
        tuner = HyperparameterTuner()
        model = ModelSelector().select_model('random_forest')
        best_model = tuner.tune_hyperparameters(
            model,
            param_grid={'n_estimators': [10, 20]},
            X=X,
            y=y
        )
        assert best_model is not None