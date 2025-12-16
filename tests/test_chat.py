"""Tests for chat session management."""

from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from helios.core.chat import ChatSession
from helios.core.types import MessageRole


def create_mock_completion(content: str) -> ChatCompletion:
    """Create a mock ChatCompletion object.

    Args:
        content: The response content.

    Returns:
        A mock ChatCompletion with the given content.
    """
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=message,
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


class TestChatSession:
    """Tests for ChatSession class."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        llm = MagicMock()
        llm.generate.return_value = create_mock_completion("Test response")
        llm.generate_streaming.return_value = iter(["Test ", "response"])
        return llm

    def test_init_without_system_prompt(self, mock_llm: MagicMock) -> None:
        """Test creating session without system prompt."""
        session = ChatSession(mock_llm)

        assert session.llm == mock_llm
        assert len(session.conversation) == 0

    def test_init_with_system_prompt(self, mock_llm: MagicMock) -> None:
        """Test creating session with system prompt."""
        session = ChatSession(mock_llm, system_prompt="You are helpful")

        assert len(session.conversation) == 1
        assert session.conversation.messages[0].role == MessageRole.SYSTEM
        assert session.conversation.messages[0].content == "You are helpful"

    def test_send_message(self, mock_llm: MagicMock) -> None:
        """Test sending a message and getting response."""
        session = ChatSession(mock_llm)
        response = session.send_message("Hello")

        assert response == "Test response"
        assert len(session.conversation) == 2
        assert session.conversation.messages[0].role == MessageRole.USER
        assert session.conversation.messages[0].content == "Hello"
        assert session.conversation.messages[1].role == MessageRole.ASSISTANT
        assert session.conversation.messages[1].content == "Test response"

    def test_send_message_updates_conversation(self, mock_llm: MagicMock) -> None:
        """Test that conversation history is updated correctly."""
        session = ChatSession(mock_llm)

        session.send_message("First message")
        assert len(session.conversation) == 2

        session.send_message("Second message")
        assert len(session.conversation) == 4

        # Verify order
        assert session.conversation.messages[0].content == "First message"
        assert session.conversation.messages[1].content == "Test response"
        assert session.conversation.messages[2].content == "Second message"
        assert session.conversation.messages[3].content == "Test response"

    def test_send_message_streaming(self, mock_llm: MagicMock) -> None:
        """Test streaming message response."""
        session = ChatSession(mock_llm)
        chunks = list(session.send_message_streaming("Hello"))

        assert chunks == ["Test ", "response"]
        assert len(session.conversation) == 2
        assert session.conversation.messages[0].role == MessageRole.USER
        assert session.conversation.messages[1].role == MessageRole.ASSISTANT
        assert session.conversation.messages[1].content == "Test response"

    def test_send_message_streaming_builds_full_response(self, mock_llm: MagicMock) -> None:
        """Test that streaming correctly builds full response."""
        mock_llm.generate_streaming.return_value = iter(["Hello", " ", "world", "!"])
        session = ChatSession(mock_llm)

        chunks = list(session.send_message_streaming("Test"))

        assert chunks == ["Hello", " ", "world", "!"]
        # Check that full response was added to history
        assert session.conversation.messages[1].content == "Hello world!"

    def test_clear_history_keep_system(self, mock_llm: MagicMock) -> None:
        """Test clearing history while keeping system messages."""
        session = ChatSession(mock_llm, system_prompt="You are helpful")
        session.send_message("Hello")

        assert len(session.conversation) == 3  # system + user + assistant

        session.clear_history(keep_system=True)

        assert len(session.conversation) == 1
        assert session.conversation.messages[0].role == MessageRole.SYSTEM
        assert session.conversation.messages[0].content == "You are helpful"

    def test_clear_history_remove_all(self, mock_llm: MagicMock) -> None:
        """Test clearing all history including system messages."""
        session = ChatSession(mock_llm, system_prompt="You are helpful")
        session.send_message("Hello")

        assert len(session.conversation) == 3

        session.clear_history(keep_system=False)

        assert len(session.conversation) == 0

    def test_clear_history_multiple_system_messages(self, mock_llm: MagicMock) -> None:
        """Test that clearing preserves all system messages."""
        session = ChatSession(mock_llm)
        session.conversation.add_system_message("First system message")
        session.conversation.add_system_message("Second system message")
        session.send_message("Hello")

        assert len(session.conversation) == 4  # 2 system + user + assistant

        session.clear_history(keep_system=True)

        assert len(session.conversation) == 2
        assert all(msg.role == MessageRole.SYSTEM for msg in session.conversation.messages)

    def test_get_message_count(self, mock_llm: MagicMock) -> None:
        """Test getting message count."""
        session = ChatSession(mock_llm)

        assert session.get_message_count() == 0

        session.send_message("Hello")
        assert session.get_message_count() == 2  # user + assistant

        session.send_message("Another message")
        assert session.get_message_count() == 4

    def test_get_message_count_with_system(self, mock_llm: MagicMock) -> None:
        """Test message count includes system messages."""
        session = ChatSession(mock_llm, system_prompt="System")

        assert session.get_message_count() == 1

        session.send_message("Hello")
        assert session.get_message_count() == 3  # system + user + assistant

    def test_get_history(self, mock_llm: MagicMock) -> None:
        """Test getting conversation history."""
        session = ChatSession(mock_llm, system_prompt="You are helpful")
        session.send_message("Hello")

        history = session.get_history()

        assert len(history) == 3
        assert history[0] == ("system", "You are helpful")
        assert history[1] == ("user", "Hello")
        assert history[2] == ("assistant", "Test response")

    def test_get_history_empty(self, mock_llm: MagicMock) -> None:
        """Test getting history from empty conversation."""
        session = ChatSession(mock_llm)
        history = session.get_history()

        assert history == []

    def test_multiple_conversations(self, mock_llm: MagicMock) -> None:
        """Test multiple back-and-forth messages."""
        session = ChatSession(mock_llm)

        session.send_message("First")
        session.send_message("Second")
        session.send_message("Third")

        assert session.get_message_count() == 6  # 3 user + 3 assistant
        history = session.get_history()

        assert history[0] == ("user", "First")
        assert history[1] == ("assistant", "Test response")
        assert history[2] == ("user", "Second")
        assert history[3] == ("assistant", "Test response")
        assert history[4] == ("user", "Third")
        assert history[5] == ("assistant", "Test response")
