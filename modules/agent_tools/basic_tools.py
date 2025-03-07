"""
Basic tools for AI agents.

This module provides basic tools that agents can use:
- Web search
- Calculator
- Current time/date
- Text processing tools
"""

import re
import math
import datetime
import json
from typing import Dict, List, Optional, Union
import requests
from bs4 import BeautifulSoup

from .tool_registry import ToolRegistry


# Initialize a tool registry
basic_tools = ToolRegistry()


@basic_tools.register_function
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression as a string.
        
    Returns:
        The result of evaluating the expression.
    """
    # Sanitize the expression to prevent code injection
    if not re.match(r'^[0-9+\-*/().%\s]*$', expression):
        return "Error: Invalid characters in expression. Only numbers and basic operators are allowed."
    
    try:
        # Safe eval using Python's built-in math functions
        # Provide a restricted namespace with only math operations
        namespace = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }
        
        # Replace common math functions with their namespace equivalents
        expression = re.sub(r'sqrt\(', 'sqrt(', expression)
        expression = re.sub(r'sin\(', 'sin(', expression)
        expression = re.sub(r'cos\(', 'cos(', expression)
        expression = re.sub(r'tan\(', 'tan(', expression)
        
        result = eval(expression, {"__builtins__": {}}, namespace)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@basic_tools.register_function
def get_current_datetime() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current date and time as an ISO-formatted string.
    """
    return datetime.datetime.now().isoformat()


@basic_tools.register_function
def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information (simplified implementation).
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title and snippet
    """
    # This is a simplified mock implementation
    # In a real scenario, you would use a search API like Google Custom Search API
    # or Bing Search API with proper API keys
    
    # For now, let's return a mock response
    return [
        {
            "title": f"Mock result {i} for: {query}",
            "snippet": f"This is a mock search result snippet for demonstration purposes. In a real implementation, this would connect to a search API. Search term: {query}",
            "url": f"https://example.com/result{i}"
        }
        for i in range(1, min(num_results + 1, 10))
    ]


@basic_tools.register_function
def fetch_webpage_content(url: str) -> str:
    """
    Fetch and extract the main content from a webpage.
    
    Args:
        url: URL of the webpage to fetch
        
    Returns:
        Extracted main text content from the webpage
    """
    try:
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines and join
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Truncate if too long
        if len(text) > 3000:
            text = text[:3000] + "... [content truncated]"
            
        return text
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"


@basic_tools.register_function
def text_analysis(text: str) -> Dict[str, Union[int, float, List[str]]]:
    """
    Perform basic text analysis.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    if not text:
        return {"error": "Empty text provided"}
    
    word_count = len(text.split())
    char_count = len(text)
    
    # Extract sentences (simplified)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    # Average sentence length in words
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Most common words (simplified)
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    common_words = [word for word, _ in common_words]
    
    return {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "average_sentence_length": round(avg_sentence_length, 2),
        "most_common_words": common_words
    }
