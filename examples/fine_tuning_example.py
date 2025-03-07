"""
Example script for fine-tuning the Mistral-7B model.

This script demonstrates how to:
1. Prepare a dataset for fine-tuning
2. Configure and initialize the fine-tuning process
3. Fine-tune the model
4. Save and evaluate the fine-tuned model
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Add parent directory to path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from modules.llm_models.fine_tuning import MistralFineTuner
from modules.data_processing.fine_tuning_dataset import FineTuningDatasetCreator
from modules.llm_models.mistral_model import MistralModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a sample dataset for fine-tuning demonstration."""
    # Sample conversations for agent tools and programming tasks
    sample_conversations = [
        [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in programming and technology."},
            {"role": "user", "content": "How do I create a Python virtual environment?"},
            {"role": "assistant", "content": "You can create a virtual environment using the following command:\n\n```bash\npython -m venv myenv\n```\n\nThen activate it with:\n- On Windows: `myenv\\Scripts\\activate`\n- On Unix/macOS: `source myenv/bin/activate`\n\nAfter activation, you can install packages with pip, and they'll be isolated to this environment."}
        ],
        [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in programming and technology."},
            {"role": "user", "content": "Write a function to find the factorial of a number."},
            {"role": "assistant", "content": "Here's a Python function to calculate the factorial of a number:\n\n```python\ndef factorial(n):\n    \"\"\"\n    Calculate the factorial of a non-negative integer n.\n    \n    Args:\n        n: A non-negative integer\n        \n    Returns:\n        The factorial of n (n!)\n    \"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    \n    if n == 0 or n == 1:\n        return 1\n    \n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    \n    return result\n```\n\nThis implementation uses an iterative approach. You could also implement it recursively:\n\n```python\ndef factorial_recursive(n):\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    \n    if n == 0 or n == 1:\n        return 1\n    \n    return n * factorial_recursive(n - 1)\n```\n\nThe iterative version is generally more efficient as it avoids the overhead of recursive function calls."}
        ],
        [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in programming and technology."},
            {"role": "user", "content": "What's the current date?"},
            {"role": "assistant", "content": "I'll help you get the current date using Python. Here's a simple code example:\n\n```python\nimport datetime\n\ncurrent_date = datetime.date.today()\nprint(f\"Today's date is: {current_date}\")\n\n# For a formatted date\nformatted_date = current_date.strftime(\"%B %d, %Y\")\nprint(f\"Formatted date: {formatted_date}\")\n```\n\nWhen you run this code, it will output the current date in both the default format (YYYY-MM-DD) and a more human-readable format."}
        ],
        [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in programming and technology."},
            {"role": "user", "content": "Calculate 15 * 27 + 42"},
            {"role": "assistant", "content": "I'll calculate 15 * 27 + 42 step by step:\n\n1. First, I'll calculate 15 * 27:\n   15 * 27 = 405\n\n2. Then, I'll add 42 to the result:\n   405 + 42 = 447\n\nSo, 15 * 27 + 42 = 447"}
        ],
        [
            {"role": "system", "content": "You are a helpful AI assistant with expertise in programming and technology."},
            {"role": "user", "content": "How can I optimize database queries for better performance?"},
            {"role": "assistant", "content": "Here are key strategies to optimize database queries for better performance:\n\n1. **Use proper indexing**:\n   - Create indexes on columns frequently used in WHERE, JOIN, and ORDER BY clauses\n   - Avoid over-indexing as it slows down INSERT/UPDATE operations\n\n2. **Write efficient queries**:\n   - Select only needed columns instead of using SELECT *\n   - Use specific conditions in WHERE clauses\n   - Avoid functions on indexed columns in WHERE clauses\n\n3. **Optimize JOINs**:\n   - Join on indexed columns\n   - Use INNER JOIN instead of OUTER when possible\n   - Minimize the number of joins in a single query\n\n4. **Use query caching**:\n   - Implement application-level caching for frequently accessed data\n   - Configure database query cache appropriately\n\n5. **Pagination and limiting**:\n   - Implement pagination for large result sets\n   - Use LIMIT to restrict results when appropriate\n\n6. **Optimize schema design**:\n   - Normalize to reduce redundancy, but consider strategic denormalization for performance\n   - Use appropriate data types and column sizes\n\n7. **Regular maintenance**:\n   - Rebuild indexes periodically\n   - Update statistics for the query optimizer\n   - Perform regular database vacuuming/cleanup\n\n8. **Analyze and profile queries**:\n   - Use EXPLAIN to understand query execution plans\n   - Monitor slow query logs\n   - Use profiling tools to identify bottlenecks\n\n9. **Consider database-specific optimizations**:\n   - PostgreSQL: Partial indexes, materialized views\n   - MySQL: Query cache, engine-specific optimizations\n   - MongoDB: Compound indexes, covered queries\n\nImplementing these strategies should significantly improve your database performance."}
        ]
    ]
    
    # Create the dataset
    creator = FineTuningDatasetCreator()
    dataset = creator.create_dataset_from_conversations(sample_conversations)
    
    return dataset


def main():
    """Run the fine-tuning example."""
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Fine-tuning will be VERY slow! Consider using a GPU for this task.")
    
    # Create directories for datasets and models
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    
    dataset_dir = data_dir / "fine_tuning_datasets"
    output_dir = models_dir / "fine_tuned_mistral"
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample dataset
    logger.info("Creating sample dataset for fine-tuning")
    dataset = create_sample_dataset()
    
    # Split dataset
    logger.info("Splitting dataset into train and test sets")
    creator = FineTuningDatasetCreator()
    dataset_split = creator.split_dataset(dataset, train_size=0.8)
    
    # Save datasets
    logger.info("Saving datasets")
    creator.save_dataset(dataset_split["train"], str(dataset_dir), "train")
    creator.save_dataset(dataset_split["test"], str(dataset_dir), "test")
    
    # Initialize fine-tuner
    logger.info("Initializing fine-tuner")
    fine_tuner = MistralFineTuner(
        base_model_name="mistralai/Mistral-7B-v0.1",
        output_dir=str(output_dir),
        use_4bit=True,
        use_lora=True
    )
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer")
    fine_tuner.load_model_and_tokenizer()
    
    # Inform user about resources
    logger.info("\n*** RESOURCE REQUIREMENTS WARNING ***")
    logger.info("Fine-tuning a 7B parameter model requires significant computational resources:")
    logger.info("- Recommended: GPU with at least 16GB VRAM")
    logger.info("- Even with 4-bit quantization and LoRA, you may still need 8GB+ VRAM")
    logger.info("- The process may take several hours to complete")
    logger.info("- Ensure you have sufficient disk space for model checkpoints")
    logger.info("*** PROCEED WITH CAUTION ***\n")
    
    # Fine-tune model
    logger.info("Starting fine-tuning")
    try:
        # Using minimal settings to demonstrate the process
        # In a real scenario, you would use more epochs and larger batch sizes
        results = fine_tuner.fine_tune(
            train_dataset=dataset_split["train"],
            eval_dataset=dataset_split["test"],
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=5,
            eval_steps=50,
            save_steps=100
        )
        
        logger.info("Fine-tuning complete!")
        
        # Save fine-tuning results
        import json
        with open(output_dir / "fine_tuning_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Test the fine-tuned model
        logger.info("Testing the fine-tuned model")
        fine_tuned_model = MistralModel(model_path=str(output_dir))
        fine_tuned_model.load_model()
        
        test_prompt = "Explain how to optimize a database query."
        logger.info(f"Test prompt: {test_prompt}")
        
        response = fine_tuned_model.generate_text(
            prompt=test_prompt,
            max_new_tokens=500,
            temperature=0.7
        )
        
        logger.info(f"Model response: {response}")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        logger.exception(e)


if __name__ == "__main__":
    main()
