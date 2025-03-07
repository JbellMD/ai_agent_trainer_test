"""
Mistral-7B-v0.1 Model Implementation for AI Agent Trainer.

This module provides implementation for the Mistral-7B-v0.1 model
using transformers library.
"""

import os
import torch
from typing import Dict, List, Any, Union
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer
from datasets import Dataset

from .llm_base import BaseLLM


class MistralModel(BaseLLM):
    """
    Implementation of Mistral-7B-v0.1 using Hugging Face transformers.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: str = "auto",
        use_quantization: bool = True,
        lora_config: Dict = None,
        logger=None,
    ):
        """
        Initialize Mistral model.
        
        Args:
            model_name: The name or path of the model to load
            device: Device to use (auto, cuda, cpu)
            use_quantization: Whether to use quantization (for memory efficiency)
            lora_config: Configuration for LoRA if fine-tuning
            logger: Logger instance
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.use_quantization = use_quantization
        self.lora_config = lora_config
        self.logger = logger or logging.getLogger(__name__)
        
        # These will be initialized when load_model is called
        self.model = None
        self.tokenizer = None
    
    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use based on availability."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, model_path: str = None, **kwargs):
        """
        Load the Mistral-7B-v0.1 model.
        
        Args:
            model_path: Optional path to a saved model.
            **kwargs: Additional parameters for model loading.
        """
        path_to_use = model_path or self.model_name
        self.logger.info(f"Loading model from {path_to_use}")
        
        # Configure quantization if needed
        quantization_config = None
        if self.use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            path_to_use,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load the model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path_to_use,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True,
                **kwargs
            )
            
            # Apply LoRA for efficient fine-tuning if config is provided
            if self.lora_config and self.device == "cuda":
                self.logger.info("Applying LoRA for parameter-efficient fine-tuning")
                lora_config = LoraConfig(**self.lora_config)
                self.model = prepare_model_for_kbit_training(self.model)
                self.model = get_peft_model(self.model, lora_config)
            
            self.logger.info("Model loaded successfully")
            return self
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text as string
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Default generation parameters
        default_params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "num_return_sequences": 1,
            "do_sample": True,
        }
        
        # Update with user-provided parameters
        params = {**default_params, **kwargs}
        
        self.logger.info("Generating text...")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **params
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
        
        return result
    
    def train(self, training_data: Union[List[Dict], Dataset], **kwargs):
        """
        Fine-tune the model with provided training data.
        
        Args:
            training_data: Either a list of dictionaries with 'input' and 'output' keys
                           or a Hugging Face dataset.
            **kwargs: Additional training parameters.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default training parameters
        default_params = {
            "output_dir": "./mistral-agent-trainer",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.001,
            "warmup_steps": 10,
            "logging_steps": 10,
            "save_steps": 50,
            "max_seq_length": 512,
        }
        
        # Update with user-provided parameters
        params = {**default_params, **kwargs}
        
        # Convert list of examples to Dataset if needed
        if isinstance(training_data, list):
            self.logger.info("Converting training data to Dataset format")
            
            # Format the data into prompt-response pairs
            formatted_data = []
            for item in training_data:
                if "input" in item and "output" in item:
                    formatted_data.append({
                        "text": f"{item['input']}\n{item['output']}"
                    })
            
            training_data = Dataset.from_dict({"text": [item["text"] for item in formatted_data]})
        
        self.logger.info("Starting fine-tuning")
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=params["output_dir"],
            num_train_epochs=params["num_train_epochs"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            warmup_steps=params["warmup_steps"],
            logging_steps=params["logging_steps"],
            save_steps=params["save_steps"],
            fp16=self.device == "cuda",
        )
        
        # Use SFTTrainer for supervised fine-tuning
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=training_data,
            tokenizer=self.tokenizer,
            max_seq_length=params["max_seq_length"],
            dataset_text_field="text" if "text" in training_data.column_names else None,
        )
        
        # Train the model
        trainer.train()
        
        # Save model and tokenizer
        if params.get("save_model", True):
            output_dir = params["output_dir"]
            self.logger.info(f"Saving fine-tuned model to {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        
        return self
    
    def evaluate(self, eval_data: Any) -> Dict[str, float]:
        """
        Evaluate model performance on provided data.
        
        Args:
            eval_data: Evaluation data.
            
        Returns:
            Dictionary of performance metrics.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Implement appropriate evaluation metrics for LLMs
        # This is a simplified implementation - in practice, you'd want to use
        # metrics like ROUGE, BLEU, perplexity, etc.
        
        metrics = {
            "perplexity": 0.0,  # Placeholder
            "completion_length": 0.0,  # Placeholder
            "response_time": 0.0,  # Placeholder
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, save_path: str) -> str:
        """
        Save the model to the specified path.
        
        Args:
            save_path: Directory to save the model.
            
        Returns:
            Path where the model was saved.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        os.makedirs(save_path, exist_ok=True)
        
        self.logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        return save_path
