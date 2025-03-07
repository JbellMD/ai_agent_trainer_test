"""
Fine-tuning utilities for large language models.

This module provides functions and utilities for fine-tuning LLMs,
particularly focused on the Mistral-7B model.
"""

import os
import logging
import torch
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json

from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

from ..data_processing.fine_tuning_dataset import FineTuningDatasetCreator

logger = logging.getLogger(__name__)


class MistralFineTuner:
    """
    Utility for fine-tuning Mistral-7B models.
    
    This class provides methods for fine-tuning the Mistral-7B model
    using various techniques like LoRA for efficient fine-tuning.
    """
    
    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "./fine_tuned_model",
        use_4bit: bool = True,
        use_lora: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            base_model_name: Hugging Face model name or path
            output_dir: Directory to save fine-tuned model
            use_4bit: Whether to use 4-bit quantization
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            device_map: Device mapping strategy for model loading
        """
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.lora_config = None
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization parameters if using 4-bit
        bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        
        if self.use_4bit:
            logger.info("Using 4-bit quantization")
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16 if bf16 else torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        else:
            quantization_config = {}
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=self.device_map,
            trust_remote_code=True,
            **quantization_config
        )
        
        # Prepare model for k-bit training if using quantization
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA if enabled
        if self.use_lora:
            logger.info("Applying LoRA adapter")
            self._apply_lora()
        
        return self.model, self.tokenizer
    
    def _apply_lora(self, lora_alpha: int = 16, lora_dropout: float = 0.1, r: int = 8):
        """
        Apply LoRA adapter to the model.
        
        Args:
            lora_alpha: Alpha parameter for LoRA
            lora_dropout: Dropout rate for LoRA
            r: Rank for LoRA
        """
        # Configure LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        max_length: int = 1024
    ) -> Dataset:
        """
        Preprocess and tokenize the dataset.
        
        Args:
            dataset: Dataset to preprocess
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call load_model_and_tokenizer first.")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing dataset",
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def prepare_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.001,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        evaluation_strategy: str = "steps",
        save_strategy: str = "steps",
        eval_steps: int = 100,
        save_steps: int = 100,
        max_grad_norm: float = 0.3,
        fp16: bool = True,
        bf16: Optional[bool] = None,
        **kwargs
    ) -> Trainer:
        """
        Prepare the trainer for fine-tuning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency in steps
            evaluation_strategy: Evaluation strategy
            save_strategy: Save strategy
            eval_steps: Evaluation frequency in steps
            save_steps: Save frequency in steps
            max_grad_norm: Maximum gradient norm
            fp16: Whether to use fp16 precision
            bf16: Whether to use bf16 precision
            **kwargs: Additional arguments for TrainingArguments
            
        Returns:
            Configured Trainer
        """
        # Determine if we can use bf16 (bfloat16) precision
        if bf16 is None:
            # Automatically detect if bf16 is supported
            bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        
        # Adjust fp16 based on bf16
        if bf16:
            fp16 = False
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            evaluation_strategy=evaluation_strategy if eval_dataset else "no",
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            max_grad_norm=max_grad_norm,
            fp16=fp16,
            bf16=bf16,
            # Enable gradient checkpointing for memory efficiency
            gradient_checkpointing=True,
            # Save total limit to prevent disk overflow
            save_total_limit=3,
            # Set proper pad token ID
            pad_token_id=self.tokenizer.pad_token_id,
            # Other settings
            **kwargs
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Not using masked language modeling
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        return trainer
    
    def fine_tune(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        **trainer_kwargs
    ):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            **trainer_kwargs: Additional arguments for prepare_trainer
            
        Returns:
            Fine-tuning results
        """
        # Make sure model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Preprocess datasets
        logger.info("Preprocessing training dataset")
        train_dataset = self.preprocess_dataset(train_dataset)
        
        if eval_dataset is not None:
            logger.info("Preprocessing evaluation dataset")
            eval_dataset = self.preprocess_dataset(eval_dataset)
        
        # Prepare trainer
        logger.info("Setting up trainer")
        trainer = self.prepare_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **trainer_kwargs
        )
        
        # Train the model
        logger.info("Starting fine-tuning")
        trainer.train()
        
        # Save the fine-tuned model
        self.save_model()
        
        # Return training results
        return trainer.state.log_history
    
    def save_model(self, custom_output_dir: Optional[str] = None):
        """
        Save the fine-tuned model.
        
        Args:
            custom_output_dir: Optional custom output directory
        """
        output_dir = custom_output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            "base_model": self.base_model_name,
            "use_4bit": self.use_4bit,
            "use_lora": self.use_lora
        }
        
        if self.use_lora and self.lora_config:
            config["lora_config"] = self.lora_config.to_dict()
        
        with open(os.path.join(output_dir, "fine_tuning_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info("Model saved successfully")


# Example usage in a script:
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import the package
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Fine-tuning will be VERY slow!")
    
    # Create sample dataset
    logger.info("Creating sample dataset")
    
    sample_conversations = [
        [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in programming and technology."},
            {"role": "user", "content": "How do I implement a binary search tree in Python?"},
            {"role": "assistant", "content": "Here's how you can implement a binary search tree in Python:\n\n```python\nclass Node:\n    def __init__(self, key):\n        self.key = key\n        self.left = None\n        self.right = None\n\nclass BinarySearchTree:\n    def __init__(self):\n        self.root = None\n    \n    def insert(self, key):\n        self.root = self._insert(self.root, key)\n    \n    def _insert(self, root, key):\n        if root is None:\n            return Node(key)\n        \n        if key < root.key:\n            root.left = self._insert(root.left, key)\n        elif key > root.key:\n            root.right = self._insert(root.right, key)\n        \n        return root\n    \n    def search(self, key):\n        return self._search(self.root, key)\n    \n    def _search(self, root, key):\n        if root is None or root.key == key:\n            return root\n        \n        if key < root.key:\n            return self._search(root.left, key)\n        \n        return self._search(root.right, key)\n```\n\nThis implementation includes the basic structure and operations of a binary search tree: insertion and search. You can extend it with additional operations like deletion, traversal, etc."}
        ],
        [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in programming and technology."},
            {"role": "user", "content": "What's the difference between a list and a tuple in Python?"},
            {"role": "assistant", "content": "In Python, lists and tuples are both sequence data types that can store collections of items, but they have several key differences:\n\n1. **Mutability**:\n   - Lists are mutable – you can modify, add, or remove elements after creation\n   - Tuples are immutable – once created, you cannot change their content\n\n2. **Syntax**:\n   - Lists use square brackets: `my_list = [1, 2, 3]`\n   - Tuples use parentheses: `my_tuple = (1, 2, 3)`\n\n3. **Performance**:\n   - Tuples are slightly faster than lists for iteration and lookup\n   - Tuples consume less memory than lists\n\n4. **Use cases**:\n   - Lists: when you need a collection that might change over time\n   - Tuples: when you want to ensure data doesn't change, like coordinates or database records\n\n5. **Methods**:\n   - Lists have more built-in methods (append, extend, insert, remove, etc.)\n   - Tuples have fewer methods due to their immutability\n\n6. **Hashability**:\n   - Tuples can be used as dictionary keys (if they contain only hashable elements)\n   - Lists cannot be used as dictionary keys\n\nChoose between them based on whether you need mutability and the specific operations you'll perform on the data."}
        ]
    ]
    
    # Create dataset creator and dataset
    creator = FineTuningDatasetCreator()
    dataset = creator.create_dataset_from_conversations(sample_conversations)
    
    # Split dataset
    dataset_split = creator.split_dataset(dataset, train_size=0.8)
    
    # Initialize fine-tuner
    fine_tuner = MistralFineTuner(
        output_dir="./fine_tuned_mistral"
    )
    
    # Load model and tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # Fine-tune model (this is a minimal example that won't actually train well)
    results = fine_tuner.fine_tune(
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1
    )
    
    logger.info("Fine-tuning complete!")
