"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from helios.utils.config import Settings, get_settings, load_settings, reset_settings


class TestSettings:
    """Tests for Settings model."""

    def test_settings_with_all_fields(self) -> None:
        """Test creating settings with all fields provided."""
        settings = Settings(
            openrouter_api_key="test-key",
            default_model="test-model",
            max_tokens=1024,
            temperature=0.5,
            app_name="test-app",
            site_url="https://test.com",
            log_level="DEBUG",
        )

        assert settings.openrouter_api_key == "test-key"
        assert settings.default_model == "test-model"
        assert settings.max_tokens == 1024
        assert settings.temperature == 0.5
        assert settings.app_name == "test-app"
        assert settings.site_url == "https://test.com"
        assert settings.log_level == "DEBUG"

    def test_settings_with_defaults(self) -> None:
        """Test that settings use default values when not provided."""
        settings = Settings(openrouter_api_key="test-key")

        assert settings.default_model == "google/gemini-flash-1.5"
        assert settings.max_tokens == 2048
        assert settings.temperature == 0.7
        assert settings.app_name == "helios"
        assert settings.site_url is None
        assert settings.log_level == "INFO"

    def test_settings_requires_api_key(self) -> None:
        """Test that openrouter_api_key is required."""
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("openrouter_api_key",) for error in errors)

    def test_temperature_validation_min(self) -> None:
        """Test that temperature must be >= 0.0."""
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test-key", temperature=-0.1)

    def test_temperature_validation_max(self) -> None:
        """Test that temperature must be <= 2.0."""
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test-key", temperature=2.1)

    def test_temperature_validation_valid_range(self) -> None:
        """Test that temperature accepts valid values in range."""
        # Min boundary
        settings = Settings(openrouter_api_key="test-key", temperature=0.0)
        assert settings.temperature == 0.0

        # Max boundary
        settings = Settings(openrouter_api_key="test-key", temperature=2.0)
        assert settings.temperature == 2.0

        # Middle value
        settings = Settings(openrouter_api_key="test-key", temperature=1.0)
        assert settings.temperature == 1.0

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-test-key"}, clear=True)
    def test_settings_from_environment(self) -> None:
        """Test loading settings from environment variables."""
        settings = Settings()
        assert settings.openrouter_api_key == "env-test-key"

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "env-key",
            "DEFAULT_MODEL": "env-model",
            "MAX_TOKENS": "4096",
            "TEMPERATURE": "0.9",
            "APP_NAME": "env-app",
            "SITE_URL": "https://env.com",
            "LOG_LEVEL": "WARNING",
        },
        clear=True,
    )
    def test_settings_all_from_environment(self) -> None:
        """Test loading all settings from environment variables."""
        settings = Settings()

        assert settings.openrouter_api_key == "env-key"
        assert settings.default_model == "env-model"
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.9
        assert settings.app_name == "env-app"
        assert settings.site_url == "https://env.com"
        assert settings.log_level == "WARNING"

    @patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "env-key", "max_tokens": "8192"},  # lowercase
        clear=True,
    )
    def test_settings_case_insensitive(self) -> None:
        """Test that environment variables are case insensitive."""
        settings = Settings()
        # Should work because case_sensitive=False in model_config
        assert settings.max_tokens == 8192

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "UNKNOWN_VAR": "value"}, clear=True)
    def test_settings_ignores_extra_env_vars(self) -> None:
        """Test that unknown environment variables are ignored."""
        # Should not raise error because extra="ignore" in model_config
        settings = Settings()
        assert settings.openrouter_api_key == "test-key"


class TestGetSettings:
    """Tests for get_settings function."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_get_settings(self) -> None:
        """Test get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.openrouter_api_key == "test-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_settings_missing_required(self) -> None:
        """Test get_settings raises error when required fields missing."""
        with pytest.raises(ValidationError):
            get_settings()


class TestLoadSettings:
    """Tests for load_settings singleton function."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_load_settings_returns_settings(self) -> None:
        """Test load_settings returns Settings instance."""
        # Reset the singleton
        reset_settings()

        settings = load_settings()
        assert isinstance(settings, Settings)
        assert settings.openrouter_api_key == "test-key"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_load_settings_caches_instance(self) -> None:
        """Test that load_settings returns cached instance."""
        reset_settings()

        settings1 = load_settings()
        settings2 = load_settings()

        # Should return the same instance
        assert settings1 is settings2

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_load_settings_singleton_behavior(self) -> None:
        """Test that modifying returned settings affects subsequent calls."""
        reset_settings()

        settings1 = load_settings()
        # Modify a field
        settings1.app_name = "modified-app"

        settings2 = load_settings()
        # Should see the modification because it's the same instance
        assert settings2.app_name == "modified-app"
