"""
LLM-based Agent implementation.

This module provides a complete implementation of an agent based on
Large Language Models that can use tools and maintain memory.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import datetime

from ..llm_models.llm_base import BaseLLM
from ..agent_tools.tool_registry import ToolRegistry
from .base_agent import BaseAgent


class Memory:
    """Memory system for LLMAgent."""
    
    def __init__(self, max_conversation_history: int = 10):
        self.conversation_history = []
        self.max_conversation_history = max_conversation_history
        self.long_term_memories = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim if exceeding max history
        if len(self.conversation_history) > self.max_conversation_history:
            self.conversation_history.pop(0)
    
    def add_long_term_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a long-term memory."""
        self.long_term_memories.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.conversation_history
    
    def get_long_term_memories(self, filter_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Get long-term memories, optionally filtered."""
        if filter_func:
            return [m for m in self.long_term_memories if filter_func(m)]
        return self.long_term_memories
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def clear_long_term_memories(self) -> None:
        """Clear long-term memories."""
        self.long_term_memories = []


class LLMAgent(BaseAgent):
    """
    Agent implementation using Large Language Models.
    
    This agent can:
    1. Process user inputs using an LLM
    2. Maintain conversation history
    3. Use tools to perform actions
    4. Store and retrieve memories
    """
    
    def __init__(
        self, 
        llm: BaseLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_conversation_history: int = 10,
        logger=None
    ):
        """
        Initialize the LLM agent.
        
        Args:
            llm: The language model to use
            tool_registry: Optional registry of tools the agent can use
            system_prompt: System prompt defining agent behavior
            max_conversation_history: Maximum number of conversation turns to keep
            logger: Logger instance
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.memory = Memory(max_conversation_history=max_conversation_history)
        self.logger = logger or logging.getLogger(__name__)
        
        # Set default system prompt if none provided
        self.system_prompt = system_prompt or self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        default_prompt = """You are a helpful AI assistant. 
Your goal is to provide accurate, helpful, and concise responses to user queries.

When responding:
1. Be truthful and provide factual information.
2. If you don't know something, admit it rather than making up information.
3. Follow user instructions carefully.
4. Be concise while still being helpful and thorough.
5. Respond in a friendly and professional tone.
"""
        
        # If tools are available, add information about them
        if self.tool_registry and self.tool_registry.list_tools():
            tool_schemas = self.tool_registry.get_tool_schemas()
            
            tools_prompt = "\nYou have access to the following tools:\n\n"
            for i, tool in enumerate(tool_schemas, 1):
                tools_prompt += f"{i}. {tool['name']}: {tool['description']}\n"
            
            tools_prompt += """
When you need to use a tool:
1. Think carefully about which tool to use based on the user's request.
2. Call the tool using the format: 
   ```tool
   {"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
   ```
3. Wait for the tool response, then use it to construct your final answer.
"""
            default_prompt += tools_prompt
        
        return default_prompt
    
    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        Build messages for LLM input including system prompt and history.
        
        Args:
            user_input: Current user input
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        for msg in self.memory.get_conversation_history():
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            List of parsed tool calls
        """
        # Pattern to match tool calls: ```tool ... ```
        pattern = r"```tool\s*(.*?)\s*```"
        matches = re.finditer(pattern, text, re.DOTALL)
        
        tool_calls = []
        for match in matches:
            tool_json = match.group(1).strip()
            try:
                tool_call = json.loads(tool_json)
                tool_calls.append({
                    "tool_call": tool_call,
                    "original_text": match.group(0)
                })
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse tool call: {tool_json}")
                continue
                
        return tool_calls
    
    def _replace_tool_calls_with_results(
        self, 
        text: str, 
        tool_calls: List[Dict[str, Any]], 
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Replace tool calls in the text with their results.
        
        Args:
            text: Original LLM response text
            tool_calls: List of extracted tool calls
            results: List of tool execution results
            
        Returns:
            Text with tool calls replaced by results
        """
        modified_text = text
        
        # Replace each tool call with its result
        for i, tool_call in enumerate(tool_calls):
            original_text = tool_call["original_text"]
            result = results[i] if i < len(results) else {"error": "No result available"}
            
            # Format the result as a code block
            result_text = f"```\n{json.dumps(result, indent=2)}\n```"
            
            # Replace the original tool call with the result
            modified_text = modified_text.replace(original_text, result_text)
            
        return modified_text
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute extracted tool calls.
        
        Args:
            tool_calls: List of extracted tool calls
            
        Returns:
            List of execution results
        """
        if not self.tool_registry:
            return [{"error": "No tool registry available"}] * len(tool_calls)
            
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_data = tool_call["tool_call"]
                tool_name = tool_data.get("name")
                tool_args = tool_data.get("arguments", {})
                
                if not tool_name:
                    results.append({"error": "Missing tool name"})
                    continue
                    
                # Execute the tool
                tool_result = self.tool_registry.execute_tool(tool_name, **tool_args)
                results.append({"result": tool_result})
                
            except Exception as e:
                self.logger.error(f"Error executing tool: {e}")
                results.append({"error": str(e)})
                
        return results
    
    def _handle_tool_calls(self, response: str) -> str:
        """
        Handle any tool calls in the LLM response.
        
        Args:
            response: LLM response that may contain tool calls
            
        Returns:
            Response with tool calls replaced by results
        """
        # Extract tool calls
        tool_calls = self._extract_tool_calls(response)
        
        if not tool_calls:
            return response
            
        # Execute tool calls
        self.logger.info(f"Executing {len(tool_calls)} tool calls")
        results = self._execute_tool_calls(tool_calls)
        
        # Replace tool calls with results
        modified_response = self._replace_tool_calls_with_results(response, tool_calls, results)
        
        # If there are still tool calls in the modified response, handle them recursively
        if "```tool" in modified_response:
            return self._handle_tool_calls(modified_response)
            
        return modified_response
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's message
            
        Returns:
            Agent's response
        """
        # Add user input to memory
        self.memory.add_message("user", user_input)
        
        # Build messages for LLM
        messages = self._build_messages(user_input)
        
        # Convert messages to a single prompt for our model
        # This is a simplification - in practice you'd use a proper chat format
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_prompt += f"{role}: {content}\n\n"
        
        formatted_prompt += "Assistant: "
        
        # Generate response from LLM
        self.logger.info("Generating response from LLM")
        model_output = self.llm.generate(formatted_prompt)
        
        # Handle any tool calls in the response
        if self.tool_registry and "```tool" in model_output:
            self.logger.info("Found tool calls in response, handling them")
            final_response = self._handle_tool_calls(model_output)
        else:
            final_response = model_output
            
        # Add assistant response to memory
        self.memory.add_message("assistant", final_response)
        
        return final_response
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add content to the agent's long-term memory.
        
        Args:
            content: Content to remember
            metadata: Optional metadata about the content
        """
        self.memory.add_long_term_memory(content, metadata)
    
    def get_memory(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve content from the agent's memory.
        
        Args:
            query: Optional query to filter memories
            
        Returns:
            List of matching memories
        """
        if query:
            # Simple keyword matching filter
            def filter_func(memory):
                return query.lower() in memory["content"].lower()
            
            return self.memory.get_long_term_memories(filter_func)
        else:
            return self.memory.get_long_term_memories()
    
    def clear_memory(self) -> None:
        """Clear the agent's conversation history."""
        self.memory.clear_conversation_history()
    
    def clear_all_memory(self) -> None:
        """Clear all of the agent's memory, including long-term memories."""
        self.memory.clear_conversation_history()
        self.memory.clear_long_term_memories()
