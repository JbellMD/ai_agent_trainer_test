"""
Tool Registry for AI Agents.

This module provides a framework for registering and managing tools
that can be used by AI agents to interact with external systems.
"""

import inspect
import json
from typing import Dict, List, Callable, Any, Optional, Union
from pydantic import BaseModel, Field


class Tool(BaseModel):
    """Definition of a tool that an agent can use."""
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    function: Callable = Field(..., description="The function to execute")
    schema: Dict[str, Any] = Field(
        default_factory=dict, 
        description="JSON schema for the tool's arguments"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.schema and self.function:
            # Generate schema from function signature
            self.schema = self._generate_schema_from_function()
    
    def _generate_schema_from_function(self) -> Dict[str, Any]:
        """Generate a JSON schema from function signature."""
        sig = inspect.signature(self.function)
        parameters = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_info = {
                "type": "string",  # Default type
                "description": f"Parameter {name}"
            }
            
            # Try to determine type from annotation if available
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_info["type"] = "string"
                elif param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                elif param.annotation == list or param.annotation == List:
                    param_info["type"] = "array"
                    param_info["items"] = {"type": "string"}
                elif param.annotation == dict or param.annotation == Dict:
                    param_info["type"] = "object"
            
            # Check if parameter has a default value
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            else:
                # This parameter is required
                pass
                
            parameters[name] = param_info
        
        return {
            "type": "object",
            "properties": parameters,
            "required": [
                name for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty and name != 'self'
            ]
        }
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with provided arguments."""
        return self.function(**kwargs)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def register_function(
        self, 
        func: Callable, 
        name: Optional[str] = None, 
        description: Optional[str] = None
    ) -> Tool:
        """Register a function as a tool."""
        if name is None:
            name = func.__name__
            
        if description is None:
            description = func.__doc__ or f"Tool for {name}"
            
        tool = Tool(
            name=name,
            description=description,
            function=func
        )
        
        self.register_tool(tool)
        return tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools, suitable for LLM function calling."""
        return [tool.to_json_schema() for tool in self.tools.values()]
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with provided arguments."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return tool.execute(**kwargs)
    
    def parse_and_execute_tool_call(self, tool_call: str) -> Dict[str, Any]:
        """
        Parse a JSON tool call from an LLM and execute it.
        
        Args:
            tool_call: JSON string with format {"name": "tool_name", "arguments": {...}}
            
        Returns:
            Dictionary with execution results
        """
        try:
            parsed = json.loads(tool_call)
            tool_name = parsed.get("name")
            arguments = parsed.get("arguments", {})
            
            if not tool_name:
                return {"error": "Missing tool name in tool call"}
                
            result = self.execute_tool(tool_name, **arguments)
            return {"result": result}
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in tool call"}
        except Exception as e:
            return {"error": str(e)}
