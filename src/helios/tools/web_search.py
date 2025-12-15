"""Web search tool using DuckDuckGo."""

from typing import Any

from duckduckgo_search import DDGS

from helios.tools.base import Tool, ToolParameter, ToolResult


class WebSearchTool(Tool):
    """A tool for searching the web using DuckDuckGo.

    Performs web searches and returns relevant results with titles,
    snippets, and URLs.
    """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "web_search"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Searches the web using DuckDuckGo and returns relevant results. "
            "Returns titles, snippets, and URLs for the top search results. "
            "Useful for finding current information, facts, or resources online."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        """Return tool parameters."""
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results to return (default: 5, max: 10)",
                required=False,
            ),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the web search tool.

        Args:
            **kwargs: Must contain 'query', optionally 'max_results'.

        Returns:
            ToolResult with search results or error.
        """
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: query",
            )

        if not isinstance(query, str):
            return ToolResult(
                success=False,
                output="",
                error="Query must be a string",
            )

        # Validate and cap max_results
        try:
            max_results = int(max_results)
            max_results = min(max(1, max_results), 10)  # Clamp between 1 and 10
        except (ValueError, TypeError):
            max_results = 5

        try:
            # Perform the search
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return ToolResult(
                    success=True,
                    output="No results found for the query.",
                )

            # Format results as readable text
            output_lines = [f"Search results for: {query}\n"]

            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                snippet = result.get("body", "No description")
                url = result.get("href", "")

                output_lines.append(f"{i}. {title}")
                output_lines.append(f"   {snippet}")
                if url:
                    output_lines.append(f"   URL: {url}")
                output_lines.append("")  # Blank line between results

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {e}",
            )
