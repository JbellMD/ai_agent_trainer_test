"""
Example script for using Mistral-7B-v0.1 with AI Agent Trainer framework.

This script demonstrates how to:
1. Load the Mistral-7B-v0.1 model
2. Generate text with the model
3. Prepare data for fine-tuning
4. Fine-tune the model
5. Save and reload the fine-tuned model
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from modules.llm_models.mistral_model import MistralModel
from modules.model_selection.model_selector import ModelSelector
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_example():
    """Example of generating text with Mistral-7B-v0.1"""
    logger.info("Loading Mistral-7B-v0.1 model")
    
    # Method 1: Load directly
    mistral = MistralModel(use_quantization=True)
    mistral.load_model()
    
    # Method 2: Load through ModelSelector (recommended for framework integration)
    # model_selector = ModelSelector()
    # mistral = model_selector.select_model('mistral-7b')
    # mistral.load_model()
    
    # Generate text
    prompt = "You are an AI agent. Your task is to help users with their coding questions. For example:"
    generated_text = mistral.generate(prompt, max_new_tokens=256)
    
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Generated: {generated_text}")


def finetune_example():
    """Example of fine-tuning Mistral-7B-v0.1 for agent tasks"""
    logger.info("Setting up for fine-tuning Mistral-7B-v0.1")
    
    # Configure LoRA for efficient fine-tuning
    lora_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
    
    # Initialize model with LoRA configuration
    mistral = MistralModel(
        use_quantization=True,
        lora_config=lora_config
    )
    mistral.load_model()
    
    # Prepare some example training data
    # In a real scenario, you would load your own dataset
    training_data = [
        {
            "input": "User: How do I create a Python virtual environment?",
            "output": "Agent: You can create a Python virtual environment using the following command:\n\n```bash\npython -m venv myenv\n```\n\nThen activate it with:\n- On Windows: `myenv\\Scripts\\activate`\n- On Unix/macOS: `source myenv/bin/activate`"
        },
        {
            "input": "User: Explain how to use Git for version control.",
            "output": "Agent: Git is a distributed version control system. Here are the basic commands:\n\n```bash\n# Initialize a repository\ngit init\n\n# Add files to staging area\ngit add .\n\n# Commit changes\ngit commit -m \"Your message\"\n\n# Create and checkout a new branch\ngit checkout -b new-branch\n\n# Push to remote repository\ngit push origin branch-name\n```"
        },
        # Add more examples as needed
    ]
    
    # Convert to Dataset format
    formatted_data = []
    for item in training_data:
        formatted_data.append({
            "text": f"{item['input']}\n{item['output']}"
        })
    
    dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_data]})
    
    # Fine-tune the model
    # Note: In a real scenario, you would want to split data into train/validation
    # and use more examples
    mistral.train(
        dataset,
        output_dir="./mistral-agent-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=512
    )
    
    # Test the fine-tuned model
    prompt = "User: How do I debug a Python program?"
    generated_text = mistral.generate(prompt, max_new_tokens=256)
    
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Generated response: {generated_text}")
    
    # Save the model
    save_path = "saved-mistral-agent"
    mistral.save(save_path)
    logger.info(f"Model saved to {save_path}")


def main():
    """Main function to run examples"""
    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available, using CPU. This will be very slow for LLM operations.")
        
    # Comment/uncomment the example you want to run
    generate_example()
    # finetune_example()  # Requires more memory and computation


if __name__ == "__main__":
    main()
