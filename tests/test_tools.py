"""Tests for agent tools."""

from unittest.mock import MagicMock, patch

import pytest

from helios.tools.base import ToolParameter, ToolSchema
from helios.tools.calculator import CalculatorTool
from helios.tools.datetime_tool import DateTimeTool
from helios.tools.registry import ToolRegistry, create_default_registry
from helios.tools.web_search import WebSearchTool


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_to_openai_format_basic(self) -> None:
        """Test converting schema to OpenAI format."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="First parameter",
                    required=True,
                )
            ],
        )

        result = schema.to_openai_format()

        assert result["type"] == "function"
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["description"] == "A test tool"
        assert "param1" in result["function"]["parameters"]["properties"]
        assert result["function"]["parameters"]["required"] == ["param1"]

    def test_to_openai_format_with_enum(self) -> None:
        """Test schema with enum parameter."""
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters=[
                ToolParameter(
                    name="format",
                    type="string",
                    description="Format type",
                    required=False,
                    enum=["json", "xml", "yaml"],
                )
            ],
        )

        result = schema.to_openai_format()
        props = result["function"]["parameters"]["properties"]

        assert props["format"]["enum"] == ["json", "xml", "yaml"]
        assert result["function"]["parameters"]["required"] == []


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    @pytest.fixture
    def calculator(self) -> CalculatorTool:
        """Create a calculator tool instance."""
        return CalculatorTool()

    def test_tool_properties(self, calculator: CalculatorTool) -> None:
        """Test tool name and description."""
        assert calculator.name == "calculator"
        assert "mathematical" in calculator.description.lower()
        assert len(calculator.parameters) == 1
        assert calculator.parameters[0].name == "expression"

    def test_simple_addition(self, calculator: CalculatorTool) -> None:
        """Test simple addition."""
        result = calculator.execute(expression="2 + 2")

        assert result.success is True
        assert result.output == "4"
        assert result.error is None

    def test_complex_expression(self, calculator: CalculatorTool) -> None:
        """Test complex mathematical expression."""
        result = calculator.execute(expression="(10 + 5) * 2 - 8")

        assert result.success is True
        assert result.output == "22"

    def test_exponentiation(self, calculator: CalculatorTool) -> None:
        """Test exponentiation."""
        result = calculator.execute(expression="2 ** 8")

        assert result.success is True
        assert result.output == "256"

    def test_builtin_functions(self, calculator: CalculatorTool) -> None:
        """Test allowed built-in functions."""
        result = calculator.execute(expression="abs(-5)")
        assert result.success is True
        assert result.output == "5"

        result = calculator.execute(expression="max(3, 7, 2)")
        assert result.success is True
        assert result.output == "7"

    def test_division_by_zero(self, calculator: CalculatorTool) -> None:
        """Test division by zero error."""
        result = calculator.execute(expression="10 / 0")

        assert result.success is False
        assert result.error is not None
        assert "zero" in result.error.lower()

    def test_invalid_syntax(self, calculator: CalculatorTool) -> None:
        """Test invalid expression syntax."""
        result = calculator.execute(expression="2 + * 2")

        assert result.success is False
        assert result.error is not None
        assert "syntax" in result.error.lower()

    def test_missing_expression(self, calculator: CalculatorTool) -> None:
        """Test missing expression parameter."""
        result = calculator.execute()

        assert result.success is False
        assert result.error is not None
        assert "missing" in result.error.lower()

    def test_invalid_expression_type(self, calculator: CalculatorTool) -> None:
        """Test non-string expression."""
        result = calculator.execute(expression=123)

        assert result.success is False
        assert result.error is not None
        assert "string" in result.error.lower()

    def test_get_schema(self, calculator: CalculatorTool) -> None:
        """Test schema generation."""
        schema = calculator.get_schema()

        assert schema.name == "calculator"
        assert len(schema.parameters) == 1


class TestDateTimeTool:
    """Tests for DateTimeTool."""

    @pytest.fixture
    def datetime_tool(self) -> DateTimeTool:
        """Create a datetime tool instance."""
        return DateTimeTool()

    def test_tool_properties(self, datetime_tool: DateTimeTool) -> None:
        """Test tool name and description."""
        assert datetime_tool.name == "datetime"
        assert "date" in datetime_tool.description.lower()
        assert len(datetime_tool.parameters) == 1
        assert datetime_tool.parameters[0].name == "format"

    def test_default_format(self, datetime_tool: DateTimeTool) -> None:
        """Test default human-readable format."""
        result = datetime_tool.execute()

        assert result.success is True
        assert result.error is None
        # Output should contain day name and month
        assert any(
            day in result.output
            for day in [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )

    def test_iso_format(self, datetime_tool: DateTimeTool) -> None:
        """Test ISO format output."""
        result = datetime_tool.execute(format="iso")

        assert result.success is True
        # ISO format contains 'T' separator
        assert "T" in result.output

    def test_timestamp_format(self, datetime_tool: DateTimeTool) -> None:
        """Test timestamp format."""
        result = datetime_tool.execute(format="timestamp")

        assert result.success is True
        # Should be a number (as string)
        assert result.output.isdigit()

    def test_full_format(self, datetime_tool: DateTimeTool) -> None:
        """Test full format with multiple fields."""
        result = datetime_tool.execute(format="full")

        assert result.success is True
        assert "Date:" in result.output
        assert "Time:" in result.output
        assert "Weekday:" in result.output
        assert "Timestamp:" in result.output

    def test_get_schema(self, datetime_tool: DateTimeTool) -> None:
        """Test schema generation."""
        schema = datetime_tool.get_schema()

        assert schema.name == "datetime"
        assert schema.parameters[0].enum == ["iso", "human", "timestamp", "full"]


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    @pytest.fixture
    def search_tool(self) -> WebSearchTool:
        """Create a web search tool instance."""
        return WebSearchTool()

    def test_tool_properties(self, search_tool: WebSearchTool) -> None:
        """Test tool name and description."""
        assert search_tool.name == "web_search"
        assert "search" in search_tool.description.lower()
        assert len(search_tool.parameters) == 2
        param_names = [p.name for p in search_tool.parameters]
        assert "query" in param_names
        assert "max_results" in param_names

    @patch("helios.tools.web_search.DDGS")
    def test_successful_search(self, mock_ddgs: MagicMock, search_tool: WebSearchTool) -> None:
        """Test successful web search."""
        # Mock search results
        mock_results = [
            {
                "title": "Test Result 1",
                "body": "This is a test description",
                "href": "https://example.com/1",
            },
            {
                "title": "Test Result 2",
                "body": "Another test description",
                "href": "https://example.com/2",
            },
        ]

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

        result = search_tool.execute(query="test query")

        assert result.success is True
        assert "Test Result 1" in result.output
        assert "Test Result 2" in result.output
        assert "https://example.com/1" in result.output

    @patch("helios.tools.web_search.DDGS")
    def test_no_results(self, mock_ddgs: MagicMock, search_tool: WebSearchTool) -> None:
        """Test search with no results."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

        result = search_tool.execute(query="nonexistent query")

        assert result.success is True
        assert "no results" in result.output.lower()

    def test_missing_query(self, search_tool: WebSearchTool) -> None:
        """Test missing query parameter."""
        result = search_tool.execute()

        assert result.success is False
        assert result.error is not None
        assert "missing" in result.error.lower()

    def test_invalid_query_type(self, search_tool: WebSearchTool) -> None:
        """Test non-string query."""
        result = search_tool.execute(query=123)

        assert result.success is False
        assert result.error is not None
        assert "string" in result.error.lower()

    @patch("helios.tools.web_search.DDGS")
    def test_max_results_limit(self, mock_ddgs: MagicMock, search_tool: WebSearchTool) -> None:
        """Test max_results parameter limits."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

        # Should cap at 10
        search_tool.execute(query="test", max_results=100)
        mock_ddgs_instance.text.assert_called_with("test", max_results=10)

        # Should not go below 1
        search_tool.execute(query="test", max_results=-5)
        mock_ddgs_instance.text.assert_called_with("test", max_results=1)


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create an empty tool registry."""
        return ToolRegistry()

    @pytest.fixture
    def calculator(self) -> CalculatorTool:
        """Create a calculator tool."""
        return CalculatorTool()

    def test_register_tool(self, registry: ToolRegistry, calculator: CalculatorTool) -> None:
        """Test registering a tool."""
        registry.register(calculator)

        assert len(registry) == 1
        assert "calculator" in registry

    def test_register_duplicate_tool(
        self, registry: ToolRegistry, calculator: CalculatorTool
    ) -> None:
        """Test registering duplicate tool raises error."""
        registry.register(calculator)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(calculator)

    def test_unregister_tool(self, registry: ToolRegistry, calculator: CalculatorTool) -> None:
        """Test unregistering a tool."""
        registry.register(calculator)
        assert "calculator" in registry

        registry.unregister("calculator")
        assert "calculator" not in registry

    def test_unregister_nonexistent_tool(self, registry: ToolRegistry) -> None:
        """Test unregistering nonexistent tool raises error."""
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_tool(self, registry: ToolRegistry, calculator: CalculatorTool) -> None:
        """Test retrieving a tool."""
        registry.register(calculator)

        tool = registry.get_tool("calculator")
        assert tool is calculator

    def test_get_nonexistent_tool(self, registry: ToolRegistry) -> None:
        """Test getting nonexistent tool returns None."""
        tool = registry.get_tool("nonexistent")
        assert tool is None

    def test_list_tools(self, registry: ToolRegistry, calculator: CalculatorTool) -> None:
        """Test listing registered tools."""
        assert registry.list_tools() == []

        registry.register(calculator)
        assert registry.list_tools() == ["calculator"]

    def test_get_schemas(self, registry: ToolRegistry, calculator: CalculatorTool) -> None:
        """Test getting schemas for all tools."""
        registry.register(calculator)

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "calculator"

    def test_execute_tool_success(self, registry: ToolRegistry, calculator: CalculatorTool) -> None:
        """Test executing a tool successfully."""
        registry.register(calculator)

        result = registry.execute_tool("calculator", expression="2 + 2")
        assert result.success is True
        assert result.output == "4"

    def test_execute_nonexistent_tool(self, registry: ToolRegistry) -> None:
        """Test executing nonexistent tool."""
        result = registry.execute_tool("nonexistent", param="value")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_create_default_registry(self) -> None:
        """Test creating registry with default tools."""
        registry = create_default_registry()

        assert len(registry) == 3
        assert "calculator" in registry
        assert "datetime" in registry
        assert "web_search" in registry
