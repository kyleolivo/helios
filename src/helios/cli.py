"""Command-line interface for Helios AI agent."""

import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from helios.core.chat import ChatSession
from helios.core.llm import create_llm
from helios.tools.registry import create_default_registry
from helios.utils.config import load_settings

console = Console()


def print_welcome() -> None:
    """Print welcome message."""
    welcome_text = """
# Welcome to Helios

An AI agent built from scratch to understand agent fundamentals.

**Commands:**
- `/exit` or `/quit` - Exit the chat
- `/clear` - Clear conversation history
- `/help` - Show this help message
- `Ctrl+C` - Exit the chat

**Available Tools:**
- `calculator` - Perform mathematical calculations
- `datetime` - Get current date and time
- `web_search` - Search the web using DuckDuckGo

Type your message and press Enter to chat!
"""
    console.print(Markdown(welcome_text))
    console.print()


def print_help() -> None:
    """Print help message."""
    help_text = """
**Available Commands:**
- `/exit` or `/quit` - Exit the chat
- `/clear` - Clear conversation history
- `/help` - Show this help message

**Tips:**
- Press `Ctrl+C` to exit anytime
- Conversation history is maintained during the session
- Use `/clear` to start fresh while keeping your session
"""
    console.print(Panel(Markdown(help_text), title="Help", border_style="blue"))


@click.group()
def main() -> None:
    """Helios - An AI agent for learning agent fundamentals."""
    pass


@main.command()
@click.option(
    "--model",
    "-m",
    help="LLM model to use (overrides config default)",
    type=str,
)
@click.option(
    "--system-prompt",
    "-s",
    help="System prompt to set conversation context",
    type=str,
)
def chat(model: str | None, system_prompt: str | None) -> None:
    """Start an interactive chat session with the AI agent.

    Args:
        model: Optional model to use (overrides config).
        system_prompt: Optional system prompt for conversation context.
    """
    try:
        # Load settings and create LLM
        settings = load_settings()
        llm = create_llm(settings, model=model)

        # Create tool registry with default tools
        tool_registry = create_default_registry()

        # Create chat session with tools
        default_system_prompt = (
            "You are Helios, a helpful AI assistant with access to tools. "
            "You are knowledgeable, concise, and friendly. "
            "When you need to perform calculations, get the current time, or search for information, "
            "use the available tools."
        )
        session = ChatSession(
            llm,
            system_prompt=system_prompt or default_system_prompt,
            tool_registry=tool_registry,
        )

        # Print welcome message
        print_welcome()

        # Main chat loop
        while True:
            try:
                # Get user input
                console.print("[bold blue]You:[/bold blue]")
                user_input = console.input()

                # Handle empty input
                if not user_input.strip():
                    continue

                # Handle special commands
                if user_input.lower() in ["/exit", "/quit"]:
                    console.print("\n[yellow]Goodbye! 👋[/yellow]\n")
                    break

                if user_input.lower() == "/clear":
                    session.clear_history(keep_system=True)
                    console.print("[green]✓ Conversation history cleared[/green]\n")
                    continue

                if user_input.lower() == "/help":
                    print_help()
                    console.print()
                    continue

                # Send message and get response (with tool calling support)
                with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                    response = session.send_message(user_input)

                # Display response with markdown formatting
                console.print("[bold green]Assistant:[/bold green]")
                if response:
                    console.print(Markdown(response))
                else:
                    console.print("[dim]No response[/dim]")
                console.print()  # Extra line for spacing

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Goodbye! 👋[/yellow]\n")
                break

            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]\n")
                continue

    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]\n")
        sys.exit(0)

    except Exception as e:
        console.print(f"[red]Failed to start chat: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
