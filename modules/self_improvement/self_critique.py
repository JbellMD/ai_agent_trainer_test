"""
Self-critique system for LLM-based agents.

This module implements mechanisms for AI agents to analyze their own responses,
identify areas for improvement, and incorporate this feedback into their learning.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class SelfCritiqueSystem:
    """
    A system that enables models to analyze and critique their own outputs.
    
    This system implements a feedback loop where the model evaluates its own
    responses, generates constructive critiques, and produces improved responses.
    These critiques can then be used for further fine-tuning.
    """
    
    def __init__(
        self,
        model,
        evaluation_dataset=None,
        critique_temperature: float = 0.7,
        improvement_temperature: float = 0.8,
        max_iterations: int = 3
    ):
        """
        Initialize the self-critique system.
        
        Args:
            model: The LLM model used for generation and critique
            evaluation_dataset: Optional dataset for evaluation
            critique_temperature: Temperature for generating critiques
            improvement_temperature: Temperature for generating improved responses
            max_iterations: Maximum number of critique-improvement iterations
        """
        self.model = model
        self.evaluation_dataset = evaluation_dataset
        self.critique_temperature = critique_temperature
        self.improvement_temperature = improvement_temperature
        self.max_iterations = max_iterations
        
        # History of critiques and improvements
        self.critique_history = []
        
        # Templates for different critique aspects
        self.critique_templates = {
            "general": (
                "Analyze the following response to the query. Identify strengths, "
                "weaknesses, and suggest specific improvements:\n"
                "Query: {query}\nResponse: {response}\n\n"
                "Critique and suggestions for improvement:"
            ),
            "factual_accuracy": (
                "Evaluate the factual accuracy of this response:\n"
                "Query: {query}\nResponse: {response}\n\n"
                "Identify any factual errors and suggest corrections:"
            ),
            "completeness": (
                "Evaluate how completely this response addresses the query:\n"
                "Query: {query}\nResponse: {response}\n\n"
                "Are there aspects of the query that weren't addressed? Explain:"
            ),
            "reasoning": (
                "Analyze the reasoning process in this response:\n"
                "Query: {query}\nResponse: {response}\n\n"
                "Identify any logical fallacies, incorrect steps, or areas where the reasoning could be improved:"
            ),
            "bias": (
                "Analyze this response for potential bias or one-sided presentation:\n"
                "Query: {query}\nResponse: {response}\n\n"
                "Identify any bias and suggest a more balanced approach:"
            )
        }
        
        # Template for improvement based on critique
        self.improvement_template = (
            "Based on the following critique, provide an improved response to the original query.\n\n"
            "Original query: {query}\n"
            "Original response: {response}\n"
            "Critique: {critique}\n\n"
            "Improved response:"
        )
    
    def critique_response(
        self,
        query: str,
        response: str,
        aspects: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate critiques for a response across multiple aspects.
        
        Args:
            query: The original query
            response: The response to critique
            aspects: Specific aspects to critique. If None, uses "general"
            
        Returns:
            Dictionary with critique results
        """
        if aspects is None:
            aspects = ["general"]
        
        critiques = {}
        
        for aspect in aspects:
            if aspect not in self.critique_templates:
                logger.warning(f"Unknown critique aspect: {aspect}. Using general template.")
                aspect = "general"
                
            template = self.critique_templates[aspect]
            prompt = template.format(query=query, response=response)
            
            # Generate critique using the model
            critique = self.model.generate_text(
                prompt=prompt,
                max_new_tokens=512,
                temperature=self.critique_temperature,
                stop_sequences=None
            )
            
            critiques[aspect] = critique.strip()
        
        # Store in history
        critique_entry = {
            "timestamp": time.time(),
            "query": query,
            "response": response,
            "critiques": critiques
        }
        self.critique_history.append(critique_entry)
        
        return critiques
    
    def generate_improved_response(
        self,
        query: str,
        response: str,
        critique: str
    ) -> str:
        """
        Generate an improved response based on critique.
        
        Args:
            query: The original query
            response: The original response
            critique: The critique of the original response
            
        Returns:
            Improved response text
        """
        prompt = self.improvement_template.format(
            query=query,
            response=response,
            critique=critique
        )
        
        # Generate improved response
        improved_response = self.model.generate_text(
            prompt=prompt,
            max_new_tokens=1024,
            temperature=self.improvement_temperature,
            stop_sequences=None
        )
        
        return improved_response.strip()
    
    def iterative_improvement(
        self,
        query: str,
        initial_response: str,
        aspects: List[str] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Iteratively improve a response through multiple rounds of critique.
        
        Args:
            query: The original query
            initial_response: The initial response
            aspects: Aspects to critique (default: ["general"])
            max_iterations: Maximum iterations (overrides instance setting if provided)
            
        Returns:
            Dictionary with improvement process and final response
        """
        if aspects is None:
            aspects = ["general"]
            
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        
        current_response = initial_response
        iteration_history = []
        
        for i in range(max_iter):
            logger.info(f"Improvement iteration {i+1}/{max_iter}")
            
            # Generate critiques
            critiques = self.critique_response(query, current_response, aspects)
            
            # Combine critiques if multiple aspects were used
            combined_critique = "\n\n".join([
                f"{aspect.upper()} CRITIQUE: {critique}" 
                for aspect, critique in critiques.items()
            ])
            
            # Generate improved response
            improved_response = self.generate_improved_response(
                query, current_response, combined_critique
            )
            
            # Track iteration
            iteration_history.append({
                "iteration": i+1,
                "response": current_response,
                "critiques": critiques,
                "improved_response": improved_response
            })
            
            # Update current response for next iteration
            current_response = improved_response
        
        return {
            "query": query,
            "initial_response": initial_response,
            "final_response": current_response,
            "iterations": iteration_history
        }
    
    def batch_evaluate_and_improve(
        self,
        queries_responses: List[Dict[str, str]],
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of query-response pairs for critique and improvement.
        
        Args:
            queries_responses: List of dicts with 'query' and 'response' keys
            save_path: Optional path to save results JSON
            
        Returns:
            List of results with critiques and improvements
        """
        results = []
        
        for i, item in enumerate(queries_responses):
            query = item["query"]
            response = item["response"]
            
            logger.info(f"Processing item {i+1}/{len(queries_responses)}")
            
            # Generate improvements
            improvement_result = self.iterative_improvement(query, response)
            results.append(improvement_result)
        
        # Save results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def extract_fine_tuning_data(
        self, 
        improvement_results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract query-response pairs for fine-tuning from improvement results.
        
        Args:
            improvement_results: Results from batch_evaluate_and_improve
            
        Returns:
            List of dicts with 'prompt' and 'completion' keys for fine-tuning
        """
        fine_tuning_data = []
        
        for result in improvement_results:
            # Add the final improved response as the target completion
            fine_tuning_data.append({
                "prompt": result["query"],
                "completion": result["final_response"]
            })
        
        return fine_tuning_data
    
    def generate_self_critique_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on the self-critique process.
        
        Returns:
            Dictionary with statistics and insights
        """
        if not self.critique_history:
            return {"error": "No critique history available"}
        
        # Calculate statistics
        num_critiques = len(self.critique_history)
        aspects_used = set()
        
        for entry in self.critique_history:
            aspects_used.update(entry["critiques"].keys())
        
        # Extract common critique patterns
        critique_texts = []
        for entry in self.critique_history:
            for critique_text in entry["critiques"].values():
                critique_texts.append(critique_text)
        
        # Simple analysis of critique texts
        avg_critique_length = sum(len(text) for text in critique_texts) / max(len(critique_texts), 1)
        
        return {
            "total_critiques": num_critiques,
            "unique_queries": len(set(entry["query"] for entry in self.critique_history)),
            "aspects_used": list(aspects_used),
            "average_critique_length": avg_critique_length,
            "timestamp": time.time()
        }


# Example usage
if __name__ == "__main__":
    # This is just placeholder code to demonstrate usage
    from modules.llm_models.mistral_model import MistralModel
    
    # Initialize model and critique system
    model = MistralModel()
    model.load_model()
    
    critique_system = SelfCritiqueSystem(model)
    
    # Example query and response
    query = "Explain how neural networks work."
    response = "Neural networks are computational models inspired by the human brain. They consist of layers of neurons that process information."
    
    # Generate critiques
    critiques = critique_system.critique_response(
        query=query,
        response=response,
        aspects=["completeness", "reasoning"]
    )
    
    # Generate improved response
    improved = critique_system.generate_improved_response(
        query=query,
        response=response,
        critique=critiques["completeness"]
    )
    
    print(f"Original response: {response}")
    print(f"\nCritique: {critiques['completeness']}")
    print(f"\nImproved response: {improved}")
