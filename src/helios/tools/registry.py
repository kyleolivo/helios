"""Tool registry for managing and accessing agent tools."""

from typing import Any

from helios.tools.base import Tool, ToolResult


class ToolRegistry:
    """Registry for managing available tools.

    Provides centralized access to tools, schema generation,
    and tool execution.
    """

    def __init__(self) -> None:
        """Initialize the tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool by name.

        Args:
            tool_name: Name of the tool to unregister.

        Raises:
            KeyError: If the tool is not registered.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve.

        Returns:
            The tool instance, or None if not found.
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-format schemas for all registered tools.

        Returns:
            List of tool schemas in OpenAI function calling format.
        """
        return [tool.get_schema().to_openai_format() for tool in self._tools.values()]

    def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with the given parameters.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Tool parameters.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self.get_tool(tool_name)

        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found",
            )

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {e}",
            )

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return tool_name in self._tools


def create_default_registry() -> ToolRegistry:
    """Create a tool registry with default tools.

    Returns:
        ToolRegistry with calculator, datetime, and web_search tools registered.
    """
    from helios.tools.calculator import CalculatorTool
    from helios.tools.datetime_tool import DateTimeTool
    from helios.tools.web_search import WebSearchTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(WebSearchTool())

    return registry
