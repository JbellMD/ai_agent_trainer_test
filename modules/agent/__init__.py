"""
Agent module for AI Agent Trainer.

This module contains implementations for different types of agents
that can be created using the framework.
"""

from .base_agent import BaseAgent
from .llm_agent import LLMAgent

__all__ = ['BaseAgent', 'LLMAgent']
