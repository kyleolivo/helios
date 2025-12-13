"""Data models for the Helios AI agent."""

from enum import Enum
from typing import Any

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
    """

    model_config = {"frozen": True}

    role: MessageRole
    content: str


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

    def to_dict(self) -> list[dict[str, str]]:
        """Convert conversation to OpenAI/Open Router API format.

        Returns:
            List of message dictionaries with 'role' and 'content' keys.
        """
        return [{"role": msg.role.value, "content": msg.content} for msg in self.messages]

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)
