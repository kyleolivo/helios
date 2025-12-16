"""Chat session management for interactive conversations."""

import json
from collections.abc import Iterator
from typing import Any

from openai.types.chat import ChatCompletion

from helios.core.llm import LLM
from helios.core.types import Conversation, Message, MessageRole
from helios.tools.registry import ToolRegistry


class ChatSession:
    """Manages an interactive chat session with an LLM.

    Handles conversation state, message history, and streaming responses.
    """

    def __init__(
        self,
        llm: LLM,
        system_prompt: str | None = None,
        tool_registry: ToolRegistry | None = None,
        max_tool_iterations: int = 5,
    ) -> None:
        """Initialize a chat session.

        Args:
            llm: The LLM client to use for generating responses.
            system_prompt: Optional system prompt to set context for the conversation.
            tool_registry: Optional registry of tools available to the agent.
            max_tool_iterations: Maximum number of tool-calling iterations to prevent loops.
        """
        self.llm = llm
        self.conversation = Conversation()
        self.tool_registry = tool_registry
        self.max_tool_iterations = max_tool_iterations

        if system_prompt:
            self.conversation.add_system_message(system_prompt)

    def send_message(self, message: str) -> str:
        """Send a message and get a response.

        Handles multi-turn tool calling automatically.

        Args:
            message: The user's message.

        Returns:
            The assistant's final response text.
        """
        self.conversation.add_user_message(message)

        # Get tool schemas if tools are available
        tools = self.tool_registry.get_schemas() if self.tool_registry else None

        # Multi-turn tool calling loop
        iterations = 0
        while iterations < self.max_tool_iterations:
            iterations += 1

            # Generate response
            response: ChatCompletion = self.llm.generate(
                self.conversation,
                tools=tools,
            )

            message_obj = response.choices[0].message

            # Check if there are tool calls
            if message_obj.tool_calls:
                # Add assistant message with tool calls to conversation
                self._add_assistant_tool_call_message(message_obj)

                # Execute each tool and add results
                for tool_call in message_obj.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    self._add_tool_result_message(tool_call.id, tool_result)

                # Continue loop to get next response
                continue
            else:
                # No tool calls, we have final response
                content = message_obj.content or ""
                self.conversation.add_assistant_message(content)
                return content

        # Max iterations reached
        final_msg = "I apologize, but I've reached the maximum number of tool calls. Please try rephrasing your request."
        self.conversation.add_assistant_message(final_msg)
        return final_msg

    def _execute_tool(self, tool_call: Any) -> str:
        """Execute a tool call and return the result.

        Args:
            tool_call: The tool call object from the LLM response.

        Returns:
            String representation of the tool result.
        """
        if not self.tool_registry:
            return "Error: No tool registry available"

        try:
            # Parse tool arguments
            args = json.loads(tool_call.function.arguments)

            # Execute tool
            result = self.tool_registry.execute_tool(tool_call.function.name, **args)

            if result.success:
                return result.output
            else:
                return f"Error: {result.error}"

        except json.JSONDecodeError as e:
            return f"Error parsing tool arguments: {e}"
        except Exception as e:
            return f"Error executing tool: {e}"

    def _add_assistant_tool_call_message(self, message: Any) -> None:
        """Add assistant message with tool calls to conversation.

        Args:
            message: The message object from ChatCompletion.
        """
        # Convert tool_calls to dict format for storage
        tool_calls_dict = None
        if message.tool_calls:
            tool_calls_dict = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        # Add message with tool_calls
        self.conversation.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=message.content or "",
                tool_calls=tool_calls_dict,
            )
        )

    def _add_tool_result_message(self, tool_call_id: str, result: str) -> None:
        """Add tool result message to conversation.

        Args:
            tool_call_id: ID of the tool call.
            result: Result from the tool execution.
        """
        # Add tool result with proper OpenAI format
        # Use SYSTEM role but with tool_call_id to indicate it's a tool result
        # The to_dict() method will format this correctly for the API
        self.conversation.messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=result,
                tool_call_id=tool_call_id,
            )
        )

    def send_message_streaming(self, message: str) -> Iterator[str]:
        """Send a message and stream the response.

        Args:
            message: The user's message.

        Yields:
            Chunks of the assistant's response as they arrive.
        """
        self.conversation.add_user_message(message)

        # Collect chunks to build full response
        chunks = []
        for chunk in self.llm.generate_streaming(self.conversation):
            chunks.append(chunk)
            yield chunk

        # Add complete response to conversation history
        full_response = "".join(chunks)
        self.conversation.add_assistant_message(full_response)

    def clear_history(self, keep_system: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_system: If True, preserves system messages. Defaults to True.
        """
        if keep_system:
            # Save system messages
            system_messages = [
                msg for msg in self.conversation.messages if msg.role == MessageRole.SYSTEM
            ]
            self.conversation.clear()
            # Restore system messages
            for msg in system_messages:
                self.conversation.messages.append(msg)
        else:
            self.conversation.clear()

    def get_message_count(self) -> int:
        """Get the number of messages in the conversation.

        Returns:
            The total number of messages (system, user, and assistant).
        """
        return len(self.conversation)

    def get_history(self) -> list[tuple[str, str]]:
        """Get conversation history as a list of (role, content) tuples.

        Returns:
            List of (role, content) tuples for all messages.
        """
        return [(msg.role.value, msg.content) for msg in self.conversation.messages]
