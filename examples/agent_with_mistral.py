"""
Example of creating an AI Agent with Mistral-7B-v0.1.

This script demonstrates how to create a simple AI agent that can:
1. Parse and understand user requests
2. Plan appropriate actions
3. Execute those actions
4. Provide responses

This is a basic framework that can be extended with more capabilities.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add parent directory to path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from modules.llm_models.mistral_model import MistralModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentMemory:
    """Simple memory for the agent to keep track of conversation history."""
    
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
    
    def add(self, role: str, content: str):
        """Add a message to history"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_formatted_history(self) -> str:
        """Format history for prompt insertion"""
        formatted = ""
        for msg in self.history:
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
        return formatted
    
    def clear(self):
        """Clear the history"""
        self.history = []


class AIAgent:
    """
    AI Agent using Mistral-7B as its core reasoning engine.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the agent.
        
        Args:
            model_path: Optional path to a fine-tuned model
        """
        self.memory = AgentMemory()
        
        # Initialize the Mistral model
        logger.info("Initializing Mistral-7B model for the agent")
        self.model = MistralModel(use_quantization=True)
        self.model.load_model(model_path)
        logger.info("Model initialized")
        
        # Define agent capabilities
        self.capabilities = [
            "answer general questions",
            "explain code",
            "generate code examples",
            "debug basic code issues"
        ]
    
    def _generate_system_prompt(self) -> str:
        """Generate the system prompt that defines agent behavior"""
        system_prompt = """You are an AI coding assistant with the following capabilities:
- Answering general programming questions
- Explaining code snippets and concepts
- Generating code examples
- Helping debug basic code issues

You should:
1. First understand the user's request
2. Think step-by-step about how to respond
3. Provide helpful, accurate, and concise responses
4. Use code blocks with appropriate syntax highlighting
5. Be honest when you don't know something

Your responses should be friendly, helpful, and focused on the user's needs.
"""
        return system_prompt
    
    def _build_full_prompt(self, user_input: str) -> str:
        """Build the full prompt with system instructions, history and user input"""
        system_prompt = self._generate_system_prompt()
        conversation_history = self.memory.get_formatted_history()
        
        full_prompt = f"{system_prompt}\n\n"
        
        if conversation_history:
            full_prompt += f"## Conversation History:\n{conversation_history}\n"
        
        full_prompt += f"User: {user_input}\n\nAssistant:"
        
        return full_prompt
    
    def _parse_response(self, model_output: str) -> str:
        """Parse and clean up the model's output"""
        # For Mistral, the output might already be well-formatted
        # But we can add additional cleaning if needed
        return model_output.strip()
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's message
            
        Returns:
            Agent's response
        """
        # Add user input to memory
        self.memory.add("user", user_input)
        
        # Build the prompt
        full_prompt = self._build_full_prompt(user_input)
        
        # Generate response
        logger.info("Generating agent response")
        model_output = self.model.generate(
            full_prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        
        # Parse and clean response
        agent_response = self._parse_response(model_output)
        
        # Add response to memory
        self.memory.add("assistant", agent_response)
        
        return agent_response


def main():
    """Run an interactive agent session"""
    # Check for GPU
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running the agent on CPU will be very slow.")
    
    print("\n=== AI Agent with Mistral-7B ===")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Initialize agent
    agent = AIAgent()
    
    # Start conversation loop
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for chatting! Goodbye.")
            break
            
        try:
            # Get agent response
            response = agent.process_input(user_input)
            print(f"\nAgent: {response}")
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            print("\nAgent: I encountered an error processing your request. Could you try again?")


if __name__ == "__main__":
    main()
