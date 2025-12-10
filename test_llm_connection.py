"""Simple script to test LLM connection via Open Router.

This script tests the basic functionality of the LLM client by sending
a simple message and receiving a response.

Before running this script:
1. Copy .env.example to .env
2. Add your Open Router API key to .env
3. Run: uv run python test_llm_connection.py
"""

from helios.core.llm import create_llm
from helios.core.types import Conversation, MessageRole
from helios.utils.config import load_settings


def main() -> None:
    """Test the LLM connection."""
    print("Testing Helios LLM Connection...")
    print("-" * 50)

    # Load settings from .env
    try:
        settings = load_settings()
        print(f"✓ Settings loaded successfully")
        print(f"  Model: {settings.default_model}")
        print(f"  Max tokens: {settings.max_tokens}")
        print(f"  Temperature: {settings.temperature}")
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        print("\nMake sure you've:")
        print("  1. Copied .env.example to .env")
        print("  2. Added your OPENROUTER_API_KEY to .env")
        return

    # Create LLM client
    llm = create_llm(settings)
    print(f"✓ LLM client created")

    # Create a simple conversation
    conversation = Conversation()
    conversation.add_user_message("Hello! Please respond with a brief greeting.")

    print(f"\n📤 Sending message: '{conversation.messages[0].content}'")
    print("-" * 50)

    # Test basic generation
    try:
        response = llm.generate(conversation)
        print(f"✓ Received response!")
        print(f"\n🤖 Assistant: {response}")
        print("-" * 50)
    except Exception as e:
        print(f"✗ Error generating response: {e}")
        return

    # Test streaming generation
    print("\n\nTesting streaming generation...")
    print("-" * 50)
    conversation.add_assistant_message(response)
    conversation.add_user_message(
        "Now tell me a very short joke about programming (max 2 sentences)."
    )

    print(f"📤 Sending message: '{conversation.messages[-1].content}'")
    print("-" * 50)

    try:
        print("🤖 Assistant (streaming): ", end="", flush=True)
        full_response = ""
        for chunk in llm.generate_streaming(conversation):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()  # New line after streaming
        print("-" * 50)
        print("\n✓ All tests passed! Your LLM connection is working.")
    except Exception as e:
        print(f"\n✗ Error with streaming: {e}")
        return


if __name__ == "__main__":
    main()
