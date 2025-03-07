"""
Evaluation module for AI Agent Trainer.

This module provides tools for evaluating the performance
of AI agents on various tasks.
"""

from .agent_evaluator import AgentEvaluator
from .metrics import calculate_metrics

__all__ = ['AgentEvaluator', 'calculate_metrics']
