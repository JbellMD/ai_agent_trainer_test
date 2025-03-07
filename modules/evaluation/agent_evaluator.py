"""
Agent evaluator for AI Agent Trainer.

This module provides a framework for systematically evaluating
the performance of AI agents on various tasks.
"""

import json
import time
import logging
import csv
import os
from typing import Dict, List, Any, Optional, Tuple, Callable

from ..agent.base_agent import BaseAgent
from .metrics import calculate_metrics, evaluate_conversation


class TestCase:
    """Represents a test case for agent evaluation."""
    
    def __init__(
        self,
        input_text: str,
        expected_output: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        validation_func: Optional[Callable[[str], bool]] = None
    ):
        """
        Initialize a test case.
        
        Args:
            input_text: The input to send to the agent
            expected_output: Expected response from the agent
            tags: List of tags categorizing this test case
            metadata: Additional metadata for the test case
            validation_func: Custom validation function for agent's output
        """
        self.input_text = input_text
        self.expected_output = expected_output
        self.tags = tags or []
        self.metadata = metadata or {}
        self.validation_func = validation_func
    
    def validate(self, agent_output: str) -> bool:
        """
        Validate the agent's output against expected output.
        
        Args:
            agent_output: The output from the agent
            
        Returns:
            True if valid, False otherwise
        """
        if self.validation_func:
            return self.validation_func(agent_output)
            
        # If no validation function or expected output, assume valid
        if not self.expected_output:
            return True
            
        # Simple string matching - in a real system, you'd want
        # more sophisticated semantic matching
        return agent_output.strip() == self.expected_output.strip()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create a TestCase from a dictionary."""
        return cls(
            input_text=data["input_text"],
            expected_output=data.get("expected_output"),
            tags=data.get("tags"),
            metadata=data.get("metadata")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TestCase to a dictionary."""
        return {
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "tags": self.tags,
            "metadata": self.metadata
        }


class TestResult:
    """Represents the result of a test case evaluation."""
    
    def __init__(
        self,
        test_case: TestCase,
        agent_output: str,
        is_valid: bool,
        metrics: Dict[str, Any],
        execution_time: float
    ):
        """
        Initialize a test result.
        
        Args:
            test_case: The test case that was evaluated
            agent_output: The output from the agent
            is_valid: Whether the output is valid
            metrics: Performance metrics
            execution_time: Time taken to execute
        """
        self.test_case = test_case
        self.agent_output = agent_output
        self.is_valid = is_valid
        self.metrics = metrics
        self.execution_time = execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TestResult to a dictionary."""
        return {
            "test_case": self.test_case.to_dict(),
            "agent_output": self.agent_output,
            "is_valid": self.is_valid,
            "metrics": self.metrics,
            "execution_time": self.execution_time
        }


class AgentEvaluator:
    """
    Framework for evaluating agents on test cases.
    
    This class provides methods for:
    1. Loading test cases
    2. Running evaluations
    3. Analyzing results
    4. Generating reports
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        logger=None
    ):
        """
        Initialize the evaluator.
        
        Args:
            agent: The agent to evaluate
            logger: Logger instance
        """
        self.agent = agent
        self.logger = logger or logging.getLogger(__name__)
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
    
    def add_test_case(self, test_case: TestCase) -> None:
        """
        Add a test case to the evaluator.
        
        Args:
            test_case: The test case to add
        """
        self.test_cases.append(test_case)
    
    def load_test_cases_from_file(self, file_path: str) -> None:
        """
        Load test cases from a JSON or CSV file.
        
        Args:
            file_path: Path to the file
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            self._load_test_cases_from_json(file_path)
        elif file_ext == '.csv':
            self._load_test_cases_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_test_cases_from_json(self, file_path: str) -> None:
        """Load test cases from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle either a list or a dictionary with a 'test_cases' key
            if isinstance(data, list):
                test_cases = data
            else:
                test_cases = data.get('test_cases', [])
                
            for tc_data in test_cases:
                self.add_test_case(TestCase.from_dict(tc_data))
                
            self.logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading test cases from {file_path}: {e}")
            raise
    
    def _load_test_cases_from_csv(self, file_path: str) -> None:
        """Load test cases from a CSV file."""
        try:
            test_cases = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Convert tags from comma-separated string to list
                    tags = row.get('tags', '').split(',') if row.get('tags') else []
                    tags = [tag.strip() for tag in tags if tag.strip()]
                    
                    # Parse metadata if present (expecting JSON format)
                    metadata = {}
                    if 'metadata' in row and row['metadata']:
                        try:
                            metadata = json.loads(row['metadata'])
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse metadata JSON: {row['metadata']}")
                    
                    test_case = TestCase(
                        input_text=row['input_text'],
                        expected_output=row.get('expected_output'),
                        tags=tags,
                        metadata=metadata
                    )
                    
                    test_cases.append(test_case)
            
            for tc in test_cases:
                self.add_test_case(tc)
                
            self.logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading test cases from {file_path}: {e}")
            raise
    
    def run_evaluation(self, clear_agent_memory: bool = True) -> List[TestResult]:
        """
        Run evaluation on all test cases.
        
        Args:
            clear_agent_memory: Whether to clear agent memory between test cases
            
        Returns:
            List of test results
        """
        self.results = []
        
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"Running test case {i+1}/{len(self.test_cases)}")
            
            if clear_agent_memory:
                self.agent.clear_memory()
            
            result = self.evaluate_test_case(test_case)
            self.results.append(result)
            
        return self.results
    
    def evaluate_test_case(self, test_case: TestCase) -> TestResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: The test case to evaluate
            
        Returns:
            Test result
        """
        start_time = time.time()
        
        try:
            # Process the input
            agent_output = self.agent.process_input(test_case.input_text)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Validate the output
            is_valid = test_case.validate(agent_output)
            
            # Calculate metrics
            metrics = calculate_metrics(
                user_input=test_case.input_text,
                agent_response=agent_output,
                expected_response=test_case.expected_output,
                response_time=execution_time
            )
            
            return TestResult(
                test_case=test_case,
                agent_output=agent_output,
                is_valid=is_valid,
                metrics=metrics,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating test case: {e}")
            
            # Return a failure result
            return TestResult(
                test_case=test_case,
                agent_output=f"ERROR: {str(e)}",
                is_valid=False,
                metrics={"error": str(e)},
                execution_time=time.time() - start_time
            )
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of evaluation results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No evaluation results available"}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.is_valid)
        
        # Calculate average execution time
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tests
        
        # Aggregate metrics across all test cases
        agg_metrics = {
            "average_input_length": sum(r.metrics.get("input_length", 0) for r in self.results) / total_tests,
            "average_response_length": sum(r.metrics.get("response_length", 0) for r in self.results) / total_tests,
            "average_response_tokens": sum(r.metrics.get("response_tokens", 0) for r in self.results) / total_tests,
        }
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "average_execution_time": avg_execution_time,
            "metrics": agg_metrics
        }
    
    def get_results_by_tag(self) -> Dict[str, Dict[str, Any]]:
        """
        Get results grouped by tags.
        
        Returns:
            Dictionary with results by tag
        """
        if not self.results:
            return {}
        
        tag_results = {}
        
        for result in self.results:
            for tag in result.test_case.tags:
                if tag not in tag_results:
                    tag_results[tag] = {
                        "total": 0,
                        "passed": 0,
                        "results": []
                    }
                
                tag_results[tag]["total"] += 1
                if result.is_valid:
                    tag_results[tag]["passed"] += 1
                tag_results[tag]["results"].append(result)
        
        # Calculate pass rates
        for tag, data in tag_results.items():
            data["pass_rate"] = data["passed"] / data["total"] if data["total"] > 0 else 0
        
        return tag_results
    
    def save_results(self, file_path: str, format: str = "json") -> None:
        """
        Save evaluation results to a file.
        
        Args:
            file_path: Path to save the results
            format: Format to save in (json or csv)
        """
        if format.lower() == "json":
            self._save_results_json(file_path)
        elif format.lower() == "csv":
            self._save_results_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_results_json(self, file_path: str) -> None:
        """Save results to a JSON file."""
        data = {
            "summary": self.get_summary_statistics(),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        self.logger.info(f"Saved evaluation results to {file_path}")
    
    def _save_results_csv(self, file_path: str) -> None:
        """Save results to a CSV file."""
        fieldnames = [
            "input_text", "expected_output", "agent_output",
            "is_valid", "execution_time", "tags"
        ]
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow({
                    "input_text": result.test_case.input_text,
                    "expected_output": result.test_case.expected_output or "",
                    "agent_output": result.agent_output,
                    "is_valid": result.is_valid,
                    "execution_time": result.execution_time,
                    "tags": ",".join(result.test_case.tags)
                })
                
        self.logger.info(f"Saved evaluation results to {file_path}")
