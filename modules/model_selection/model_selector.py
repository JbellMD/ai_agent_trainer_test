from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from ..utils.logging import AutoTrainerLogger

class ModelSelector:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        self.models = {
            'random_forest': RandomForestClassifier(),
            'logistic_regression': LogisticRegression(),
            'svm': SVC()
        }
        
        # Initialize LLM models dictionary - these will be loaded on demand
        self.llm_models = {
            'mistral-7b': {
                'class': 'MistralModel',
                'module': 'ai_agent_trainer_test.modules.llm_models.mistral_model',
                'params': {
                    'model_name': 'mistralai/Mistral-7B-v0.1',
                    'use_quantization': True
                }
            },
            'llama-2-7b': {
                'class': 'LlamaModel',
                'module': 'ai_agent_trainer_test.modules.llm_models.llama_model',
                'params': {
                    'model_name': 'meta-llama/Llama-2-7b-hf',
                    'use_quantization': True
                }
            }
        }
        
    def select_model(self, model_name: str, **kwargs):
        """Select a model by name"""
        self.logger.log(f"Selecting model: {model_name}")
        
        # Check if it's a traditional ML model
        if model_name in self.models:
            return self.models[model_name]
            
        # Check if it's an LLM model
        elif model_name in self.llm_models:
            return self._load_llm_model(model_name, **kwargs)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _load_llm_model(self, model_name: str, **kwargs):
        """Load an LLM model dynamically"""
        model_info = self.llm_models[model_name]
        
        try:
            # Dynamically import the module
            import importlib
            module = importlib.import_module(model_info['module'])
            
            # Get the model class
            model_class = getattr(module, model_info['class'])
            
            # Combine default parameters with provided parameters
            params = {**model_info['params'], **kwargs}
            
            # Create and return the model instance
            model_instance = model_class(**params)
            self.logger.log(f"Successfully loaded LLM model: {model_name}")
            return model_instance
            
        except Exception as e:
            self.logger.log(f"Error loading LLM model {model_name}: {e}", level='error')
            raise
        
    def get_available_models(self) -> list:
        """Get list of available models"""
        return list(self.models.keys()) + list(self.llm_models.keys())