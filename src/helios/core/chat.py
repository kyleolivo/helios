"""Chat session management for interactive conversations."""

from typing import Optional

from helios.core.llm import LLM
from helios.core.types import Conversation, MessageRole


class ChatSession:
    """Manages an interactive chat session with an LLM.

    Handles conversation state, message history, and streaming responses.
    """

    def __init__(self, llm: LLM, system_prompt: Optional[str] = None) -> None:
        """Initialize a chat session.

        Args:
            llm: The LLM client to use for generating responses.
            system_prompt: Optional system prompt to set context for the conversation.
        """
        self.llm = llm
        self.conversation = Conversation()

        if system_prompt:
            self.conversation.add_system_message(system_prompt)

    def send_message(self, message: str) -> str:
        """Send a message and get a response.

        Args:
            message: The user's message.

        Returns:
            The assistant's response.
        """
        self.conversation.add_user_message(message)
        response = self.llm.generate(self.conversation)
        self.conversation.add_assistant_message(response)
        return response

    def send_message_streaming(self, message: str):
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
                msg for msg in self.conversation.messages
                if msg.role == MessageRole.SYSTEM
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
        return [
            (msg.role.value, msg.content)
            for msg in self.conversation.messages
        ]
