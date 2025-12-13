"""Configuration management for Helios using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Open Router API Configuration
    openrouter_api_key: str = Field(
        ...,
        description="Open Router API key for LLM access",
    )

    # Model Configuration
    default_model: str = Field(
        default="google/gemini-flash-1.5",
        description="Default LLM model to use",
    )

    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for model responses",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for model responses (0.0-2.0)",
    )

    # Application Metadata
    app_name: str = Field(
        default="helios",
        description="Application name for Open Router tracking",
    )

    site_url: Optional[str] = Field(
        default=None,
        description="Site URL for Open Router analytics",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings (cached).

    Returns:
        Settings: Application settings loaded from environment.

    Raises:
        ValidationError: If required settings are missing or invalid.
    """
    return Settings()


def load_settings() -> Settings:
    """Load or return cached settings instance.

    This is an alias for get_settings() for backward compatibility.

    Returns:
        Settings: Cached application settings.
    """
    return get_settings()
