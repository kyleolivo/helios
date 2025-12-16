"""Data models for the Helios AI agent."""

from enum import Enum
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles for messages in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The content of the message.
        tool_calls: Optional tool calls made by assistant (for function calling).
        tool_call_id: Optional ID when this message is a tool result.
    """

    role: MessageRole
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class Conversation(BaseModel):
    """Represents a conversation with message history.

    Attributes:
        messages: List of messages in the conversation.
        metadata: Optional metadata about the conversation.
    """

    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the conversation.

        Args:
            role: The role of the message sender.
            content: The content of the message.
        """
        self.messages.append(Message(role=role, content=content))

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.

        Args:
            content: The user's message content.
        """
        self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.

        Args:
            content: The assistant's message content.
        """
        self.add_message(MessageRole.ASSISTANT, content)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation.

        Args:
            content: The system message content.
        """
        self.add_message(MessageRole.SYSTEM, content)

    def to_dict(self) -> list[ChatCompletionMessageParam]:
        """Convert conversation to OpenAI/Open Router API format.

        Returns:
            List of message dictionaries compatible with OpenAI SDK types.
        """
        messages = []
        for msg in self.messages:
            # Determine role - tool messages need role="tool"
            role = "tool" if msg.tool_call_id else msg.role.value

            message_dict: dict[str, Any] = {
                "role": role,
                "content": msg.content,
            }
            # Add optional tool-related fields if present
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            messages.append(message_dict)
        return cast(list[ChatCompletionMessageParam], messages)

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)
