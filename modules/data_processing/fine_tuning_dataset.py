"""
Fine-tuning dataset creation and management.

This module provides utilities for creating, managing, and 
processing datasets for fine-tuning large language models.
"""

import os
import json
import logging
import random
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class FineTuningDatasetCreator:
    """
    Utility for creating and managing datasets for LLM fine-tuning.
    
    This class helps with:
    1. Creating consistent prompt-completion pairs
    2. Converting data to various formats
    3. Splitting data into train/test sets
    4. Quality filtering and preprocessing
    """
    
    def __init__(self, conversation_template: Optional[str] = None):
        """
        Initialize the dataset creator.
        
        Args:
            conversation_template: Optional template for formatting conversations
        """
        self.conversation_template = conversation_template or self._default_conversation_template()
    
    def _default_conversation_template(self) -> str:
        """Get the default conversation template."""
        return """<s>[INST] {instruction} [/INST] {response}</s>"""
    
    def format_prompt_completion(
        self, 
        instruction: str, 
        response: str
    ) -> str:
        """
        Format a prompt-completion pair using the conversation template.
        
        Args:
            instruction: User instruction or prompt
            response: Model completion or response
            
        Returns:
            Formatted text for fine-tuning
        """
        return self.conversation_template.format(
            instruction=instruction,
            response=response
        )
    
    def format_conversation(
        self, 
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Format a multi-turn conversation for fine-tuning.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted conversation text
        """
        formatted_text = ""
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system" and i == 0:
                # Special handling for system prompt at the start
                formatted_text = f"<s>[INST] {content}\n\n"
            elif role == "user":
                if formatted_text and not formatted_text.endswith("[INST] "):
                    formatted_text += f"[INST] {content} [/INST] "
                else:
                    formatted_text += f"{content} [/INST] "
            elif role == "assistant":
                formatted_text += f"{content}</s>"
                if i < len(messages) - 1:  # Not the last message
                    formatted_text += "<s>"
        
        return formatted_text
    
    def create_dataset_from_conversations(
        self, 
        conversations: List[List[Dict[str, str]]]
    ) -> Dataset:
        """
        Create a dataset from conversations.
        
        Args:
            conversations: List of conversation message lists
            
        Returns:
            HuggingFace Dataset
        """
        formatted_texts = []
        
        for conversation in conversations:
            formatted_text = self.format_conversation(conversation)
            formatted_texts.append({"text": formatted_text})
        
        return Dataset.from_pandas(pd.DataFrame(formatted_texts))
    
    def create_dataset_from_prompt_completions(
        self, 
        prompt_completions: List[Dict[str, str]]
    ) -> Dataset:
        """
        Create a dataset from prompt-completion pairs.
        
        Args:
            prompt_completions: List of dictionaries with 'prompt'/'instruction' and 'completion'/'response' keys
            
        Returns:
            HuggingFace Dataset
        """
        formatted_texts = []
        
        for item in prompt_completions:
            # Support different key names
            instruction = item.get("prompt") or item.get("instruction") or ""
            response = item.get("completion") or item.get("response") or ""
            
            formatted_text = self.format_prompt_completion(instruction, response)
            formatted_texts.append({"text": formatted_text})
        
        return Dataset.from_pandas(pd.DataFrame(formatted_texts))
    
    def load_from_json(self, file_path: str) -> Dataset:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            HuggingFace Dataset
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine format based on structure
            if isinstance(data, list):
                # Check first item to determine format
                if len(data) > 0:
                    first_item = data[0]
                    
                    # Check for conversation format
                    if isinstance(first_item, list) and all(isinstance(x, dict) and "role" in x for x in first_item):
                        return self.create_dataset_from_conversations(data)
                    
                    # Check for prompt-completion format
                    if isinstance(first_item, dict) and any(key in first_item for key in ["prompt", "instruction"]):
                        return self.create_dataset_from_prompt_completions(data)
            
            # If we get here, format wasn't recognized
            raise ValueError("Unrecognized data format in JSON file")
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def load_from_jsonl(self, file_path: str) -> Dataset:
        """
        Load data from a JSONL file (one JSON object per line).
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            HuggingFace Dataset
        """
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            # Determine if this is conversations or prompt-completions
            if len(data) > 0:
                first_item = data[0]
                
                # Check for prompt-completion format (most common for JSONL)
                if isinstance(first_item, dict) and any(key in first_item for key in ["prompt", "instruction"]):
                    return self.create_dataset_from_prompt_completions(data)
            
            # If format wasn't recognized, raise error
            raise ValueError("Unrecognized data format in JSONL file")
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def load_from_csv(self, file_path: str, prompt_col: str = "prompt", completion_col: str = "completion") -> Dataset:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            prompt_col: Column name for prompts
            completion_col: Column name for completions
            
        Returns:
            HuggingFace Dataset
        """
        try:
            df = pd.read_csv(file_path)
            
            # Verify columns exist
            if prompt_col not in df.columns or completion_col not in df.columns:
                raise ValueError(f"CSV must have '{prompt_col}' and '{completion_col}' columns")
            
            # Convert to prompt-completion format
            prompt_completions = []
            for _, row in df.iterrows():
                prompt_completions.append({
                    "prompt": row[prompt_col],
                    "completion": row[completion_col]
                })
            
            return self.create_dataset_from_prompt_completions(prompt_completions)
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def load_dataset(self, file_path: str, **kwargs) -> Dataset:
        """
        Load a dataset from a file based on its extension.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            HuggingFace Dataset
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            return self.load_from_json(file_path)
        elif file_ext == '.jsonl':
            return self.load_from_jsonl(file_path)
        elif file_ext in ['.csv', '.tsv']:
            prompt_col = kwargs.get('prompt_col', 'prompt')
            completion_col = kwargs.get('completion_col', 'completion')
            return self.load_from_csv(file_path, prompt_col, completion_col)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def split_dataset(
        self, 
        dataset: Dataset, 
        train_size: float = 0.8,
        seed: int = 42
    ) -> Dict[str, Dataset]:
        """
        Split a dataset into training and test sets.
        
        Args:
            dataset: Dataset to split
            train_size: Proportion for training set
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train' and 'test' datasets
        """
        split = dataset.train_test_split(train_size=train_size, seed=seed)
        return {
            'train': split['train'],
            'test': split['test']
        }
    
    def filter_by_length(
        self, 
        dataset: Dataset,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Dataset:
        """
        Filter dataset items by text length.
        
        Args:
            dataset: Dataset to filter
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters)
            
        Returns:
            Filtered dataset
        """
        def length_filter(example):
            text_len = len(example["text"])
            
            if min_length is not None and text_len < min_length:
                return False
                
            if max_length is not None and text_len > max_length:
                return False
                
            return True
        
        return dataset.filter(length_filter)
    
    def save_dataset(
        self, 
        dataset: Dataset,
        output_dir: str,
        name: str = "fine_tuning_data"
    ) -> None:
        """
        Save a dataset to disk.
        
        Args:
            dataset: Dataset to save
            output_dir: Directory to save in
            name: Name for the dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(os.path.join(output_dir, name))
        logger.info(f"Saved dataset to {os.path.join(output_dir, name)}")


# Example usage in a script:
if __name__ == "__main__":
    # Sample data for testing
    sample_conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I create a Python virtual environment?"},
            {"role": "assistant", "content": "You can create a virtual environment using the following command:\n\npython -m venv myenv\n\nThen activate it with:\n- On Windows: myenv\\Scripts\\activate\n- On Unix/macOS: source myenv/bin/activate"}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what a decorator is in Python."},
            {"role": "assistant", "content": "A decorator in Python is a design pattern that allows you to extend the behavior of a function or method without modifying its source code. It uses the @decorator syntax and is implemented as a function that takes another function as an argument and returns a new function that usually extends the original function's behavior."}
        ]
    ]
    
    # Create dataset
    creator = FineTuningDatasetCreator()
    dataset = creator.create_dataset_from_conversations(sample_conversations)
    
    print(f"Created dataset with {len(dataset)} examples")
    print(f"Example: {dataset[0]['text'][:100]}...")
