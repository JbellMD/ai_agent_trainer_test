"""
Example script for evaluating an AI agent.

This script demonstrates how to:
1. Set up test cases for agent evaluation
2. Run the evaluation process
3. Analyze the results
4. Generate a report
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from modules.agent.llm_agent import LLMAgent
from modules.llm_models.mistral_model import MistralModel
from modules.agent_tools.tool_registry import ToolRegistry
from modules.agent_tools.basic_tools import basic_tools
from modules.evaluation.agent_evaluator import AgentEvaluator, TestCase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_cases():
    """Create sample test cases for agent evaluation."""
    test_cases = [
        TestCase(
            input_text="What is the capital of France?",
            expected_output=None,  # We'll use validation instead of exact matching
            tags=["general_knowledge", "geography"],
            validation_func=lambda output: "paris" in output.lower()
        ),
        TestCase(
            input_text="Calculate 15 * 27 + 42",
            expected_output=None,
            tags=["math", "calculation"],
            validation_func=lambda output: "447" in output
        ),
        TestCase(
            input_text="Write a function in Python to check if a string is a palindrome.",
            expected_output=None,
            tags=["coding", "python"],
            validation_func=lambda output: "def" in output and "palindrome" in output.lower()
        ),
        TestCase(
            input_text="What's the difference between a list and a tuple in Python?",
            expected_output=None,
            tags=["coding", "python", "concepts"],
            validation_func=lambda output: "mutable" in output.lower() and "immutable" in output.lower()
        ),
        TestCase(
            input_text="Explain the concept of recursion in programming.",
            expected_output=None,
            tags=["coding", "concepts"],
            validation_func=lambda output: "recursion" in output.lower() and "function" in output.lower()
        ),
        TestCase(
            input_text="How can I optimize database queries for better performance?",
            expected_output=None,
            tags=["databases", "optimization"],
            validation_func=lambda output: "index" in output.lower()
        ),
        TestCase(
            input_text="Write a regular expression to match email addresses.",
            expected_output=None,
            tags=["coding", "regex"],
            validation_func=lambda output: "@" in output and "\\." in output or "\\@" in output
        ),
        TestCase(
            input_text="What are the main components of the MERN stack?",
            expected_output=None,
            tags=["web_development", "frameworks"],
            validation_func=lambda output: all(term in output.lower() for term in ["mongo", "express", "react", "node"])
        ),
        TestCase(
            input_text="Explain how machine learning models can be deployed to production.",
            expected_output=None,
            tags=["machine_learning", "deployment"],
            validation_func=lambda output: "deploy" in output.lower() and "model" in output.lower()
        ),
        TestCase(
            input_text="What is the current date?",
            expected_output=None,
            tags=["tools", "date_time"],
            validation_func=lambda output: any(term in output.lower() for term in ["today", "date", "20", "202"])
        )
    ]
    
    return test_cases


def save_test_cases(test_cases, file_path):
    """Save test cases to a JSON file."""
    test_cases_data = [tc.to_dict() for tc in test_cases]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(test_cases_data, f, indent=2)
        
    logger.info(f"Saved {len(test_cases)} test cases to {file_path}")


def initialize_agent():
    """Initialize an LLM agent with tools."""
    # Initialize the Mistral model
    logger.info("Initializing Mistral-7B model")
    model = MistralModel(use_quantization=True)
    model.load_model()
    
    # Create an agent with tools
    agent = LLMAgent(
        llm=model,
        tool_registry=basic_tools,
        system_prompt="""You are a helpful AI coding assistant. 
Your goal is to provide accurate, helpful, and clear responses to user queries about programming, 
software development, and computer science concepts.

When appropriate, provide code examples to illustrate your explanations.
For calculations and other functions, you can use tools when they would be helpful.
"""
    )
    
    return agent


def main():
    """Run the agent evaluation example."""
    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available, using CPU. This will be very slow for LLM operations.")
    
    # Create directory for test cases if it doesn't exist
    test_cases_dir = Path(__file__).parent.parent / "data" / "test_cases"
    os.makedirs(test_cases_dir, exist_ok=True)
    
    # Path for test cases
    test_cases_path = test_cases_dir / "coding_agent_test_cases.json"
    
    # Create and save test cases if they don't exist
    if not os.path.exists(test_cases_path):
        logger.info("Creating test cases")
        test_cases = create_test_cases()
        save_test_cases(test_cases, test_cases_path)
    
    # Initialize agent
    logger.info("Initializing agent")
    agent = initialize_agent()
    
    # Create evaluator
    logger.info("Creating evaluator")
    evaluator = AgentEvaluator(agent)
    
    # Load test cases
    logger.info(f"Loading test cases from {test_cases_path}")
    evaluator.load_test_cases_from_file(str(test_cases_path))
    
    # Run evaluation
    logger.info("Running evaluation")
    evaluator.run_evaluation(clear_agent_memory=True)
    
    # Get and print summary statistics
    summary = evaluator.get_summary_statistics()
    logger.info(f"Evaluation Summary:")
    logger.info(f"Total tests: {summary['total_tests']}")
    logger.info(f"Passed tests: {summary['passed_tests']}")
    logger.info(f"Pass rate: {summary['pass_rate'] * 100:.1f}%")
    logger.info(f"Average execution time: {summary['average_execution_time']:.2f} seconds")
    
    # Results by tag
    tag_results = evaluator.get_results_by_tag()
    logger.info("\nResults by tag:")
    for tag, data in tag_results.items():
        logger.info(f"{tag}: {data['pass_rate'] * 100:.1f}% pass rate ({data['passed']}/{data['total']})")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "data" / "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = results_dir / "agent_evaluation_results.json"
    logger.info(f"Saving results to {results_path}")
    evaluator.save_results(str(results_path))
    
    results_csv_path = results_dir / "agent_evaluation_results.csv"
    logger.info(f"Saving results to {results_csv_path}")
    evaluator.save_results(str(results_csv_path), format="csv")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
