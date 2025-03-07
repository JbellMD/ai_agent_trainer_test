"""
Base LLM Class for AI Agent Trainer.

This module provides a base class for all LLM implementations
to ensure consistent interface across different models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseLLM(ABC):
    """
    Base class for Large Language Models.
    
    All LLM implementations should inherit from this class and
    implement the required methods.
    """
    
    @abstractmethod
    def load_model(self, model_path: str = None, **kwargs):
        """
        Load the model from a specified path or download it.
        
        Args:
            model_path: Optional path to a saved model.
            **kwargs: Additional parameters for model loading.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on input prompt.
        
        Args:
            prompt: The input text prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text as a string.
        """
        pass
    
    @abstractmethod
    def train(self, training_data: Any, **kwargs):
        """
        Fine-tune the model with provided training data.
        
        Args:
            training_data: Training data for fine-tuning.
            **kwargs: Additional training parameters.
        """
        pass
    
    @abstractmethod
    def evaluate(self, eval_data: Any) -> Dict[str, float]:
        """
        Evaluate model performance on provided data.
        
        Args:
            eval_data: Evaluation data.
            
        Returns:
            Dictionary of performance metrics.
        """
        pass
    
    @abstractmethod
    def save(self, save_path: str) -> str:
        """
        Save the model to a specified path.
        
        Args:
            save_path: Path where the model will be saved.
            
        Returns:
            Path where the model was saved.
        """
        pass
