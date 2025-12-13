"""Tests for LLM client abstractions."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from helios.core.llm import LLM, OpenRouterLLM, create_llm
from helios.core.types import Conversation, MessageRole
from helios.utils.config import Settings


class TestLLMAbstractClass:
    """Tests for LLM abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLM abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            LLM()

    def test_subclass_must_implement_generate(self):
        """Test that subclass must implement generate method."""

        class IncompleteLLM(LLM):
            def generate_streaming(self, conversation, max_tokens=None, temperature=None):
                pass

        with pytest.raises(TypeError):
            IncompleteLLM()

    def test_subclass_must_implement_generate_streaming(self):
        """Test that subclass must implement generate_streaming method."""

        class IncompleteLLM(LLM):
            def generate(self, conversation, max_tokens=None, temperature=None):
                return "test"

        with pytest.raises(TypeError):
            IncompleteLLM()


class TestOpenRouterLLM:
    """Tests for OpenRouterLLM implementation."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            openrouter_api_key="test-api-key",
            default_model="test-model",
            max_tokens=2048,
            temperature=0.7,
            app_name="test-app",
            site_url="https://test.com",
        )

    @pytest.fixture
    def conversation(self):
        """Create test conversation."""
        conv = Conversation()
        conv.add_system_message("You are helpful")
        conv.add_user_message("Hello")
        return conv

    def test_init_with_default_model(self, settings):
        """Test initializing OpenRouterLLM with default model."""
        llm = OpenRouterLLM(settings)
        assert llm.model == "test-model"
        assert llm.settings == settings

    def test_init_with_custom_model(self, settings):
        """Test initializing OpenRouterLLM with custom model."""
        llm = OpenRouterLLM(settings, model="custom-model")
        assert llm.model == "custom-model"
        assert llm.settings == settings

    def test_client_initialization(self, settings):
        """Test that OpenAI client is initialized with correct settings."""
        llm = OpenRouterLLM(settings)

        assert str(llm.client.base_url) == "https://openrouter.ai/api/v1/"
        assert llm.client.api_key == "test-api-key"

    @patch("helios.core.llm.OpenAI")
    def test_generate(self, mock_openai_class, settings, conversation):
        """Test generate method with mocked API response."""
        # Create mock response
        mock_message = ChatCompletionMessage(role="assistant", content="Hello there!")
        mock_choice = Choice(
            finish_reason="stop",
            index=0,
            message=mock_message,
        )
        mock_response = ChatCompletion(
            id="test-id",
            choices=[mock_choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )

        # Setup mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Test
        llm = OpenRouterLLM(settings)
        response = llm.generate(conversation)

        assert response == "Hello there!"
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=conversation.to_dict(),
            max_tokens=2048,
            temperature=0.7,
        )

    @patch("helios.core.llm.OpenAI")
    def test_generate_with_custom_params(self, mock_openai_class, settings, conversation):
        """Test generate method with custom max_tokens and temperature."""
        mock_message = ChatCompletionMessage(role="assistant", content="Response")
        mock_choice = Choice(
            finish_reason="stop",
            index=0,
            message=mock_message,
        )
        mock_response = ChatCompletion(
            id="test-id",
            choices=[mock_choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = OpenRouterLLM(settings)
        response = llm.generate(conversation, max_tokens=1024, temperature=0.5)

        assert response == "Response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=conversation.to_dict(),
            max_tokens=1024,
            temperature=0.5,
        )

    @patch("helios.core.llm.OpenAI")
    def test_generate_with_temperature_zero(self, mock_openai_class, settings, conversation):
        """Test that temperature=0 is passed correctly (not treated as None)."""
        mock_message = ChatCompletionMessage(role="assistant", content="Response")
        mock_choice = Choice(
            finish_reason="stop",
            index=0,
            message=mock_message,
        )
        mock_response = ChatCompletion(
            id="test-id",
            choices=[mock_choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = OpenRouterLLM(settings)
        llm.generate(conversation, temperature=0.0)

        # Should use 0.0, not fall back to settings.temperature
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    @patch("helios.core.llm.OpenAI")
    def test_generate_empty_response(self, mock_openai_class, settings, conversation):
        """Test generate handles empty response."""
        mock_response = ChatCompletion(
            id="test-id",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = OpenRouterLLM(settings)
        response = llm.generate(conversation)

        assert response == ""

    @patch("helios.core.llm.OpenAI")
    def test_generate_none_content(self, mock_openai_class, settings, conversation):
        """Test generate handles None content."""
        mock_message = ChatCompletionMessage(role="assistant", content=None)
        mock_choice = Choice(
            finish_reason="stop",
            index=0,
            message=mock_message,
        )
        mock_response = ChatCompletion(
            id="test-id",
            choices=[mock_choice],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = OpenRouterLLM(settings)
        response = llm.generate(conversation)

        assert response == ""

    @patch("helios.core.llm.OpenAI")
    def test_generate_streaming(self, mock_openai_class, settings, conversation):
        """Test generate_streaming method."""
        # Create mock streaming chunks
        chunk1 = ChatCompletionChunk(
            id="test-id",
            choices=[
                StreamChoice(
                    delta=ChoiceDelta(content="Hello", role="assistant"),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        chunk2 = ChatCompletionChunk(
            id="test-id",
            choices=[
                StreamChoice(
                    delta=ChoiceDelta(content=" there", role=None),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        chunk3 = ChatCompletionChunk(
            id="test-id",
            choices=[
                StreamChoice(
                    delta=ChoiceDelta(content="!", role=None),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])
        mock_openai_class.return_value = mock_client

        llm = OpenRouterLLM(settings)
        chunks = list(llm.generate_streaming(conversation))

        assert chunks == ["Hello", " there", "!"]
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=conversation.to_dict(),
            max_tokens=2048,
            temperature=0.7,
            stream=True,
        )

    @patch("helios.core.llm.OpenAI")
    def test_generate_streaming_with_custom_params(self, mock_openai_class, settings, conversation):
        """Test generate_streaming with custom parameters."""
        mock_chunk = ChatCompletionChunk(
            id="test-id",
            choices=[
                StreamChoice(
                    delta=ChoiceDelta(content="test", role="assistant"),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        mock_openai_class.return_value = mock_client

        llm = OpenRouterLLM(settings)
        list(llm.generate_streaming(conversation, max_tokens=512, temperature=0.3))

        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=conversation.to_dict(),
            max_tokens=512,
            temperature=0.3,
            stream=True,
        )

    @patch("helios.core.llm.OpenAI")
    def test_generate_streaming_skips_empty_chunks(self, mock_openai_class, settings, conversation):
        """Test that generate_streaming skips chunks without content."""
        chunk_with_content = ChatCompletionChunk(
            id="test-id",
            choices=[
                StreamChoice(
                    delta=ChoiceDelta(content="Hello", role="assistant"),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        chunk_without_content = ChatCompletionChunk(
            id="test-id",
            choices=[
                StreamChoice(
                    delta=ChoiceDelta(content=None, role=None),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(
            [chunk_with_content, chunk_without_content]
        )
        mock_openai_class.return_value = mock_client

        llm = OpenRouterLLM(settings)
        chunks = list(llm.generate_streaming(conversation))

        # Should only return the chunk with content
        assert chunks == ["Hello"]


class TestCreateLLM:
    """Tests for create_llm factory function."""

    def test_create_llm_returns_openrouter(self):
        """Test that create_llm returns OpenRouterLLM instance."""
        settings = Settings(openrouter_api_key="test-key")
        llm = create_llm(settings)

        assert isinstance(llm, OpenRouterLLM)
        assert isinstance(llm, LLM)

    def test_create_llm_with_default_model(self):
        """Test create_llm uses default model from settings."""
        settings = Settings(openrouter_api_key="test-key", default_model="default-model")
        llm = create_llm(settings)

        assert llm.model == "default-model"

    def test_create_llm_with_custom_model(self):
        """Test create_llm with custom model override."""
        settings = Settings(openrouter_api_key="test-key", default_model="default-model")
        llm = create_llm(settings, model="custom-model")

        assert llm.model == "custom-model"
