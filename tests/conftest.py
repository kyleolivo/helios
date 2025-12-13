"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def disable_dotenv():
    """Disable .env file loading during tests.

    This prevents the .env file from affecting test behavior.
    Tests should explicitly set environment variables when needed.
    """
    # Find project root and .env file
    env_file = Path(__file__).parent.parent / ".env"
    temp_file = Path(__file__).parent.parent / ".env.test_backup"

    # Temporarily rename .env if it exists
    if env_file.exists():
        env_file.rename(temp_file)

    # Clear any env vars that were loaded from .env
    env_keys_to_clear = [
        "OPENROUTER_API_KEY",
        "DEFAULT_MODEL",
        "MAX_TOKENS",
        "TEMPERATURE",
        "APP_NAME",
        "SITE_URL",
        "LOG_LEVEL",
    ]
    saved_env = {}
    for key in env_keys_to_clear:
        if key in os.environ:
            saved_env[key] = os.environ.pop(key)

    yield

    # Restore .env file
    if temp_file.exists():
        temp_file.rename(env_file)

    # Restore environment variables
    os.environ.update(saved_env)


@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """Reset the settings cache before each test.

    This ensures tests don't interfere with each other by sharing
    cached settings instances.
    """
    from helios.utils.config import get_settings

    # Clear the lru_cache to ensure fresh settings for each test
    get_settings.cache_clear()

    yield

    # Clear cache again after test
    get_settings.cache_clear()
