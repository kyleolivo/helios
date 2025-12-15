"""DateTime tool for getting current date and time information."""

from datetime import UTC, datetime
from typing import Any

from helios.tools.base import Tool, ToolParameter, ToolResult


class DateTimeTool(Tool):
    """A tool for getting current date and time information.

    Provides current date, time, and timezone information in various formats.
    """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "datetime"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Gets the current date and time. "
            "Can return in different formats: "
            "'iso' (ISO 8601 format), "
            "'human' (human-readable), "
            "'timestamp' (Unix timestamp), "
            "or 'full' (detailed information)"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        """Return tool parameters."""
        return [
            ToolParameter(
                name="format",
                type="string",
                description="Output format: iso, human, timestamp, or full",
                required=False,
                enum=["iso", "human", "timestamp", "full"],
            )
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the datetime tool.

        Args:
            **kwargs: May contain 'format' key specifying output format.

        Returns:
            ToolResult with current date/time information.
        """
        format_type = kwargs.get("format", "human")

        try:
            now = datetime.now(UTC)

            if format_type == "iso":
                output = now.isoformat()

            elif format_type == "timestamp":
                output = str(int(now.timestamp()))

            elif format_type == "full":
                output = (
                    f"Date: {now.strftime('%Y-%m-%d')}\n"
                    f"Time: {now.strftime('%H:%M:%S')} UTC\n"
                    f"Weekday: {now.strftime('%A')}\n"
                    f"Timestamp: {int(now.timestamp())}\n"
                    f"ISO: {now.isoformat()}"
                )

            else:  # human (default)
                output = now.strftime("%A, %B %d, %Y at %I:%M %p UTC")

            return ToolResult(
                success=True,
                output=output,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to get date/time: {e}",
            )
