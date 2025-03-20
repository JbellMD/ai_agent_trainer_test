# AI Agent Trainer

![CI/CD Pipeline](https://github.com/JbellMD/ai_agent_trainer/workflows/CI/CD/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

The AI Agent Trainer is a comprehensive framework for autonomous machine learning model creation and optimization.

## Features

- **Automated Model Training**: Supports multiple ML algorithms with automatic hyperparameter tuning
- **Data Processing**: Built-in tools for data cleaning, feature engineering, and preprocessing
- **Model Evaluation**: Comprehensive metrics and explainability tools
- **Deployment Ready**: Docker and Kubernetes support for production deployment
- **Monitoring**: Real-time performance tracking and alerting
- **Dashboard**: Interactive visualization of model performance and explanations
- **LLM Integration**: Support for Large Language Models including Mistral-7B for advanced AI agents

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai_agent_trainer.git
cd ai_agent_trainer

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from ai_agent_trainer.core import AutoTrainer

# Initialize trainer
trainer = AutoTrainer()

# Load and preprocess data
trainer.load_data('data/sample.csv')

# Train model
trainer.train()

# Evaluate model
metrics = trainer.evaluate()

# Start explainability dashboard
trainer.explain()
```

## Using Large Language Models

The AI Agent Trainer now supports Large Language Models (LLMs) for creating sophisticated AI agents. 
The framework currently integrates with Mistral-7B-v0.1, an open-source high-quality LLM.

### Setting up Mistral-7B

```python
from modules.model_selection.model_selector import ModelSelector

# Initialize model selector
model_selector = ModelSelector()

# Load Mistral-7B model
mistral_model = model_selector.select_model('mistral-7b')
mistral_model.load_model()

# Generate text
response = mistral_model.generate("Your prompt here")
print(response)
```

### Creating an AI Agent

A complete example of creating an AI agent using Mistral-7B is available in the examples directory:

```bash
# Run the example agent
python examples/agent_with_mistral.py
```

### Fine-tuning for Specific Tasks

You can fine-tune the model for your specific agent tasks:

```python
from modules.llm_models.mistral_model import MistralModel
from datasets import Dataset

# Prepare your fine-tuning data
training_data = Dataset.from_dict({
    "text": [
        "User: How do I create a virtual environment?\nAssistant: You can use the command: python -m venv myenv",
        # Add more examples
    ]
})

# Initialize model with LoRA for efficient fine-tuning
mistral = MistralModel(
    use_quantization=True,
    lora_config={
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
)
mistral.load_model()

# Fine-tune the model
mistral.train(
    training_data,
    output_dir="./my-agent-model",
    num_train_epochs=3
)
```

## Documentation

- [API Reference](api.md)
- [Configuration Guide](documentation/configuration.md)
- [Deployment Guide](documentation/deployment.md)
- [Developer Guide](documentation/developer.md)

## Contributing

We welcome contributions! Please see our Contributing Guide for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
