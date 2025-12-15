"""Base classes and interfaces for agent tools."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Defines a parameter for a tool.

    Attributes:
        name: The parameter name.
        type: The parameter type (string, number, boolean, etc.).
        description: Human-readable description of the parameter.
        required: Whether this parameter is required.
        enum: Optional list of allowed values.
    """

    name: str
    type: str = Field(description="Parameter type: string, number, integer, boolean, array, object")
    description: str
    required: bool = True
    enum: list[str] | None = Field(default=None, description="Allowed values")


class ToolSchema(BaseModel):
    """JSON schema representation of a tool for LLM function calling.

    This follows the OpenAI function calling schema format, which is
    compatible with most LLM providers including Open Router.

    Attributes:
        name: The tool/function name.
        description: What the tool does.
        parameters: List of parameters the tool accepts.
    """

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function calling schema format.
        """
        # Build properties dict for parameters
        properties = {}
        required = []

        for param in self.parameters:
            param_schema: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }

            if param.enum:
                param_schema["enum"] = param.enum

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolResult(BaseModel):
    """Result from executing a tool.

    Attributes:
        success: Whether the tool executed successfully.
        output: The tool's output (string representation).
        error: Error message if execution failed.
    """

    success: bool
    output: str
    error: str | None = None


class Tool(ABC):
    """Abstract base class for agent tools.

    Tools extend the agent's capabilities by providing specific functionality
    like calculations, web search, file operations, etc.

    Subclasses must implement:
    - name: Tool identifier
    - description: What the tool does
    - parameters: List of parameters the tool accepts
    - execute: The actual tool logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name (used for identification)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Return the list of parameters this tool accepts."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool parameters as keyword arguments.

        Returns:
            ToolResult containing the execution result.
        """
        pass

    def get_schema(self) -> ToolSchema:
        """Get the JSON schema for this tool.

        Returns:
            ToolSchema that can be sent to LLMs for function calling.
        """
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"
