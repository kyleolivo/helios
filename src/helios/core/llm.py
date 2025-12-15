"""LLM client abstractions for interacting with language models."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from openai import OpenAI
from openai.types.chat import ChatCompletion

from helios.core.types import Conversation
from helios.utils.config import Settings


class LLM(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        conversation: Conversation,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            conversation: The conversation history to send to the LLM.
            max_tokens: Maximum tokens for the response (overrides config).
            temperature: Temperature for sampling (overrides config).

        Returns:
            The generated response text.

        Raises:
            Exception: If the API call fails.
        """
        pass

    @abstractmethod
    def generate_streaming(
        self,
        conversation: Conversation,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Generate a response from the LLM with streaming.

        Args:
            conversation: The conversation history to send to the LLM.
            max_tokens: Maximum tokens for the response (overrides config).
            temperature: Temperature for sampling (overrides config).

        Yields:
            Chunks of the generated response text.

        Raises:
            Exception: If the API call fails.
        """
        pass


class OpenRouterLLM(LLM):
    """LLM client for Open Router API.

    Open Router provides access to multiple LLM providers through a unified API
    that's compatible with the OpenAI SDK.
    """

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
    ) -> None:
        """Initialize the Open Router LLM client.

        Args:
            settings: Application settings containing API key and defaults.
            model: Model to use (overrides settings.default_model).
        """
        self.settings = settings
        self.model = model or settings.default_model

        # Initialize OpenAI client configured for Open Router
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            default_headers={
                "HTTP-Referer": settings.site_url or "https://github.com/helios",
                "X-Title": settings.app_name,
            },
        )

    def generate(
        self,
        conversation: Conversation,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a response from the LLM via Open Router.

        Args:
            conversation: The conversation history to send to the LLM.
            max_tokens: Maximum tokens for the response (overrides config).
            temperature: Temperature for sampling (overrides config).

        Returns:
            The generated response text.

        Raises:
            Exception: If the API call fails.
        """
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=conversation.to_dict(),
            max_tokens=max_tokens or self.settings.max_tokens,
            temperature=temperature if temperature is not None else self.settings.temperature,
        )

        # Extract the response text
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content if content else ""

        return ""

    def generate_streaming(
        self,
        conversation: Conversation,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Generate a response from the LLM with streaming via Open Router.

        Args:
            conversation: The conversation history to send to the LLM.
            max_tokens: Maximum tokens for the response (overrides config).
            temperature: Temperature for sampling (overrides config).

        Yields:
            Chunks of the generated response text.

        Raises:
            Exception: If the API call fails.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=conversation.to_dict(),
            max_tokens=max_tokens or self.settings.max_tokens,
            temperature=temperature if temperature is not None else self.settings.temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content


def create_llm(settings: Settings, model: str | None = None) -> LLM:
    """Factory function to create an LLM client.

    Args:
        settings: Application settings.
        model: Model to use (overrides settings.default_model).

    Returns:
        An LLM client instance configured for Open Router.
    """
    return OpenRouterLLM(settings=settings, model=model)
