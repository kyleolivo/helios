"""Tests for core data types (Message, Conversation, MessageRole)."""

import pytest
from helios.core.types import Conversation, Message, MessageRole


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_message_role_values(self):
        """Test that MessageRole enum has correct values."""
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"

    def test_message_role_is_string(self):
        """Test that MessageRole behaves like a string."""
        role = MessageRole.USER
        assert isinstance(role, str)
        assert role == "user"

    def test_message_role_has_name_and_value(self):
        """Test that MessageRole has both name and value attributes."""
        role = MessageRole.USER
        assert role.name == "USER"
        assert role.value == "user"


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Test creating a message with valid data."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_message_with_different_roles(self):
        """Test creating messages with different roles."""
        system_msg = Message(role=MessageRole.SYSTEM, content="You are helpful")
        user_msg = Message(role=MessageRole.USER, content="Hello")
        assistant_msg = Message(role=MessageRole.ASSISTANT, content="Hi there")

        assert system_msg.role == MessageRole.SYSTEM
        assert user_msg.role == MessageRole.USER
        assert assistant_msg.role == MessageRole.ASSISTANT

    def test_message_validation_requires_content(self):
        """Test that message requires content."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Message(role=MessageRole.USER)

    def test_message_validation_requires_role(self):
        """Test that message requires role."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Message(content="Hello")

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(role=MessageRole.USER, content="Hello")
        msg_dict = msg.model_dump()
        assert msg_dict["role"] == MessageRole.USER
        assert msg_dict["content"] == "Hello"


class TestConversation:
    """Tests for Conversation model."""

    def test_create_empty_conversation(self):
        """Test creating an empty conversation."""
        conv = Conversation()
        assert len(conv) == 0
        assert conv.messages == []
        assert conv.metadata == {}

    def test_conversation_with_initial_messages(self):
        """Test creating conversation with initial messages."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        conv = Conversation(messages=messages)
        assert len(conv) == 2
        assert conv.messages == messages

    def test_add_message(self):
        """Test adding a message to conversation."""
        conv = Conversation()
        conv.add_message(MessageRole.USER, "Hello")
        assert len(conv) == 1
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[0].content == "Hello"

    def test_add_user_message(self):
        """Test add_user_message helper method."""
        conv = Conversation()
        conv.add_user_message("Hello")
        assert len(conv) == 1
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[0].content == "Hello"

    def test_add_assistant_message(self):
        """Test add_assistant_message helper method."""
        conv = Conversation()
        conv.add_assistant_message("Hi there")
        assert len(conv) == 1
        assert conv.messages[0].role == MessageRole.ASSISTANT
        assert conv.messages[0].content == "Hi there"

    def test_add_system_message(self):
        """Test add_system_message helper method."""
        conv = Conversation()
        conv.add_system_message("You are helpful")
        assert len(conv) == 1
        assert conv.messages[0].role == MessageRole.SYSTEM
        assert conv.messages[0].content == "You are helpful"

    def test_add_multiple_messages(self):
        """Test adding multiple messages in sequence."""
        conv = Conversation()
        conv.add_system_message("You are helpful")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there")

        assert len(conv) == 3
        assert conv.messages[0].role == MessageRole.SYSTEM
        assert conv.messages[1].role == MessageRole.USER
        assert conv.messages[2].role == MessageRole.ASSISTANT

    def test_to_dict(self):
        """Test converting conversation to API format."""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there")

        result = conv.to_dict()
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_to_dict_with_system_message(self):
        """Test to_dict includes system messages."""
        conv = Conversation()
        conv.add_system_message("You are helpful")
        conv.add_user_message("Hello")

        result = conv.to_dict()
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_clear(self):
        """Test clearing all messages from conversation."""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there")
        assert len(conv) == 2

        conv.clear()
        assert len(conv) == 0
        assert conv.messages == []

    def test_len_operator(self):
        """Test that len() works on conversation."""
        conv = Conversation()
        assert len(conv) == 0

        conv.add_user_message("Hello")
        assert len(conv) == 1

        conv.add_assistant_message("Hi there")
        assert len(conv) == 2

    def test_metadata(self):
        """Test conversation metadata."""
        metadata = {"user_id": "123", "session_id": "abc"}
        conv = Conversation(metadata=metadata)
        assert conv.metadata == metadata

    def test_metadata_is_independent_per_instance(self):
        """Test that each conversation has its own metadata dict."""
        conv1 = Conversation()
        conv2 = Conversation()

        conv1.metadata["key"] = "value1"
        conv2.metadata["key"] = "value2"

        assert conv1.metadata["key"] == "value1"
        assert conv2.metadata["key"] == "value2"

    def test_messages_are_independent_per_instance(self):
        """Test that each conversation has its own messages list."""
        conv1 = Conversation()
        conv2 = Conversation()

        conv1.add_user_message("Hello from conv1")
        conv2.add_user_message("Hello from conv2")

        assert len(conv1) == 1
        assert len(conv2) == 1
        assert conv1.messages[0].content == "Hello from conv1"
        assert conv2.messages[0].content == "Hello from conv2"
