"""
Base Agent class for AI Agent Trainer.

This module provides a base class for all agent implementations
to ensure consistent interface across different agent types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseAgent(ABC):
    """
    Base class for all agent implementations.
    
    All agent implementations should inherit from this class and
    implement the required methods.
    """
    
    @abstractmethod
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's message or query
            
        Returns:
            Agent's response
        """
        pass
    
    @abstractmethod
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add content to the agent's memory.
        
        Args:
            content: Content to remember
            metadata: Optional metadata about the content
        """
        pass
    
    @abstractmethod
    def get_memory(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve content from the agent's memory.
        
        Args:
            query: Optional query to filter memories
            
        Returns:
            List of matching memories
        """
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        pass
