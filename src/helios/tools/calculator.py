"""Calculator tool for mathematical operations."""

from typing import Any

from helios.tools.base import Tool, ToolParameter, ToolResult


class CalculatorTool(Tool):
    """A tool for performing mathematical calculations.

    Safely evaluates mathematical expressions using Python's eval()
    with restricted globals to prevent code execution.
    """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "calculator"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Performs mathematical calculations. "
            "Supports basic operations (+, -, *, /), "
            "exponentiation (**), and parentheses. "
            "Example: '2 + 2 * 3' or '(10 + 5) ** 2'"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        """Return tool parameters."""
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="The mathematical expression to evaluate",
                required=True,
            )
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the calculator tool.

        Args:
            **kwargs: Must contain 'expression' key with the math expression.

        Returns:
            ToolResult with the calculation result or error.
        """
        expression = kwargs.get("expression")

        if not expression:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: expression",
            )

        if not isinstance(expression, str):
            return ToolResult(
                success=False,
                output="",
                error="Expression must be a string",
            )

        try:
            # Safe evaluation: only allow math operations, no built-in functions
            # This prevents code execution like eval("__import__('os').system('ls')")
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            }

            # Evaluate the expression with restricted namespace
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return ToolResult(
                success=True,
                output=str(result),
            )

        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid expression syntax: {e}",
            )

        except ZeroDivisionError:
            return ToolResult(
                success=False,
                output="",
                error="Division by zero",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Calculation error: {e}",
            )
