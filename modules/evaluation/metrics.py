"""
Metrics for evaluating AI agent performance.

This module provides various metrics to evaluate the performance
of AI agents on different dimensions.
"""

import re
import json
import time
from typing import Dict, List, Any, Union, Optional


def calculate_response_time(start_time: float, end_time: float) -> float:
    """
    Calculate the response time in seconds.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Response time in seconds
    """
    return end_time - start_time


def count_tokens(text: str) -> int:
    """
    Count approximate number of tokens in text.
    
    This is a very rough approximation. In a real system,
    you would use the model's tokenizer for accurate counts.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    # Very rough approximation: split by whitespace and punctuation
    return len(re.findall(r'\b\w+\b|[^\w\s]', text))


def measure_tool_usage(response: str) -> Dict[str, int]:
    """
    Measure tool usage in a response.
    
    Args:
        response: Agent response text
        
    Returns:
        Dictionary with tool usage statistics
    """
    # Pattern to match tool calls: ```tool ... ```
    pattern = r"```tool\s*(.*?)\s*```"
    matches = re.finditer(pattern, response, re.DOTALL)
    
    tool_counts = {}
    total_tools = 0
    
    for match in matches:
        total_tools += 1
        tool_json = match.group(1).strip()
        
        try:
            tool_call = json.loads(tool_json)
            tool_name = tool_call.get("name", "unknown")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        except json.JSONDecodeError:
            tool_counts["unparseable"] = tool_counts.get("unparseable", 0) + 1
    
    return {
        "total_tools_used": total_tools,
        "unique_tools_used": len(tool_counts),
        "tool_distribution": tool_counts
    }


def calculate_metrics(
    user_input: str,
    agent_response: str,
    expected_response: Optional[str] = None,
    response_time: Optional[float] = None,
    tools_used: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for an agent interaction.
    
    Args:
        user_input: User's input text
        agent_response: Agent's response text
        expected_response: Optional expected response for scoring
        response_time: Optional response time in seconds
        tools_used: Optional list of tools used
        
    Returns:
        Dictionary with various performance metrics
    """
    metrics = {}
    
    # Basic response statistics
    metrics["input_length"] = len(user_input)
    metrics["response_length"] = len(agent_response)
    metrics["input_tokens"] = count_tokens(user_input)
    metrics["response_tokens"] = count_tokens(agent_response)
    
    # Responsiveness
    if response_time is not None:
        metrics["response_time_seconds"] = response_time
        metrics["tokens_per_second"] = metrics["response_tokens"] / max(response_time, 0.001)
    
    # Tool usage (if not provided, try to extract from response)
    if tools_used is None:
        metrics["tool_usage"] = measure_tool_usage(agent_response)
    else:
        metrics["tool_usage"] = {
            "total_tools_used": len(tools_used),
            "unique_tools_used": len(set(t.get("name", "") for t in tools_used)),
            "tool_distribution": {}  # Would need to compute this
        }
    
    # Content metrics
    metrics["contains_code_blocks"] = "```" in agent_response
    metrics["bullet_points"] = len(re.findall(r'^\s*[-*]\s+', agent_response, re.MULTILINE))
    metrics["numbered_lists"] = len(re.findall(r'^\s*\d+\.\s+', agent_response, re.MULTILINE))
    
    # Referenced input terms
    # This is a simple heuristic to check if response references terms from input
    input_terms = set(re.findall(r'\b\w{4,}\b', user_input.lower()))
    response_terms = set(re.findall(r'\b\w{4,}\b', agent_response.lower()))
    input_terms_in_response = input_terms.intersection(response_terms)
    
    metrics["referenced_input_terms"] = len(input_terms_in_response)
    metrics["referenced_input_terms_ratio"] = (
        len(input_terms_in_response) / max(len(input_terms), 1)
    )
    
    # Comparative metrics (if expected response provided)
    if expected_response:
        # Calculate similarity with expected response
        expected_tokens = set(re.findall(r'\b\w+\b', expected_response.lower()))
        response_tokens = set(re.findall(r'\b\w+\b', agent_response.lower()))
        
        # Jaccard similarity
        intersection = expected_tokens.intersection(response_tokens)
        union = expected_tokens.union(response_tokens)
        jaccard = len(intersection) / max(len(union), 1)
        
        metrics["expected_response_similarity"] = jaccard
    
    return metrics


def evaluate_conversation(
    conversation: List[Dict[str, str]],
    expected_responses: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Evaluate metrics for an entire conversation.
    
    Args:
        conversation: List of conversation messages
        expected_responses: Optional mapping of turn indices to expected responses
        
    Returns:
        Dictionary with aggregated metrics
    """
    metrics = {
        "turns": len(conversation) // 2,  # Assuming alternating user/assistant messages
        "total_user_tokens": 0,
        "total_assistant_tokens": 0,
        "average_response_length": 0,
        "metrics_by_turn": []
    }
    
    user_messages = [msg for msg in conversation if msg.get("role") == "user"]
    assistant_messages = [msg for msg in conversation if msg.get("role") == "assistant"]
    
    turn_metrics = []
    
    # Process each turn
    for i, (user_msg, assistant_msg) in enumerate(zip(user_messages, assistant_messages)):
        expected = expected_responses.get(i) if expected_responses else None
        
        turn_metric = calculate_metrics(
            user_input=user_msg.get("content", ""),
            agent_response=assistant_msg.get("content", ""),
            expected_response=expected
        )
        
        metrics["total_user_tokens"] += turn_metric["input_tokens"]
        metrics["total_assistant_tokens"] += turn_metric["response_tokens"]
        
        turn_metrics.append(turn_metric)
    
    # Calculate aggregated metrics
    metrics["average_response_length"] = (
        sum(m["response_length"] for m in turn_metrics) / max(len(turn_metrics), 1)
    )
    
    metrics["average_response_tokens"] = (
        metrics["total_assistant_tokens"] / max(len(turn_metrics), 1)
    )
    
    metrics["metrics_by_turn"] = turn_metrics
    
    return metrics
