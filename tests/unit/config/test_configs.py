"""
Tests for configuration module.
"""

import logging
from unittest.mock import patch

import pytest

from src.config.configs import EnvConfig, get_default_configs


class TestEnvConfig:
    """Tests for EnvConfig class."""

    def test_loads_from_environment_variables(self, mock_env_vars):
        """Test that EnvConfig loads values from environment variables."""
        config = EnvConfig()  # type: ignore

        assert config.api_key == "test-api-key"
        assert config.api_url == "https://api.test.com"
        assert config.model_name == "test-model"
        assert config.tavily_api_key == "test-tavily-key"
        assert config.langfuse_public_key == "test-public-key"
        assert config.langfuse_secret_key == "test-secret-key"  # noqa: S105
        assert config.langfuse_base_url == "http://localhost:3000"
        assert config.log_level == "DEBUG"
        assert config.environment == "development"

    def test_warning_logged_when_tavily_key_missing(self, caplog):
        """Test warning is logged when Tavily API key is not set."""
        with patch.dict(
            "os.environ",
            {
                "API_KEY": "test-key",
                "API_URL": "https://test.com",
                "MODEL_NAME": "test-model",
                "TAVILY_API_KEY": "",  # Empty key
                "LANGFUSE_PUBLIC_KEY": "test",
                "LANGFUSE_SECRET_KEY": "test",
                "LANGFUSE_BASE_URL": "http://test.com",
            },
            clear=True,
        ):
            config = EnvConfig()  # type: ignore

            # Check warning was logged
            warning_messages = [
                rec.message for rec in caplog.records if rec.levelno == logging.WARNING
            ]
            assert any("TAVILY_API_KEY not set" in msg for msg in warning_messages)

    def test_warning_logged_when_langfuse_keys_missing(self, caplog):
        """Test warning is logged when Langfuse keys are not set."""
        with patch.dict(
            "os.environ",
            {
                "API_KEY": "test-key",
                "API_URL": "https://test.com",
                "MODEL_NAME": "test-model",
                "TAVILY_API_KEY": "test-tavily",
                "LANGFUSE_PUBLIC_KEY": "",
                "LANGFUSE_SECRET_KEY": "",
                "LANGFUSE_BASE_URL": "http://test.com",
            },
            clear=True,
        ):
            config = EnvConfig()  # type: ignore

            # Check warning was logged
            warning_messages = [
                rec.message for rec in caplog.records if rec.levelno == logging.WARNING
            ]
            assert any("Langfuse keys not set" in msg for msg in warning_messages)

    def test_log_level_case_insensitive(self, mock_env_vars):
        """Test that log_level accepts different case values."""
        with patch.dict("os.environ", {"LOG_LEVEL": "debug"}):
            config = EnvConfig()  # type: ignore
            assert config.log_level == "debug"

        with patch.dict("os.environ", {"LOG_LEVEL": "WARNING"}):
            config = EnvConfig()  # type: ignore
            assert config.log_level == "WARNING"

    def test_environment_case_insensitive(self, mock_env_vars):
        """Test that environment accepts different case values."""
        with patch.dict("os.environ", {"ENVIRONMENT": "PRODUCTION"}):
            config = EnvConfig()  # type: ignore
            assert config.environment == "PRODUCTION"

        with patch.dict("os.environ", {"ENVIRONMENT": "staging"}):
            config = EnvConfig()  # type: ignore
            assert config.environment == "staging"

    def test_model_post_init_logs_configuration(self, caplog, mock_env_vars):
        """Test that model_post_init logs configuration details."""
        with caplog.at_level(logging.INFO):
            config = EnvConfig()  # type: ignore

            # Check info logs were created
            info_messages = [
                rec.message for rec in caplog.records if rec.levelno == logging.INFO
            ]
            assert any("Configuration loaded from" in msg for msg in info_messages)
            assert any(f"Model: {config.model_name}" in msg for msg in info_messages)
            assert any(f"API URL: {config.api_url}" in msg for msg in info_messages)
            assert any(
                f"Environment: {config.environment}" in msg for msg in info_messages
            )
            assert any(f"Log level: {config.log_level}" in msg for msg in info_messages)


class TestRuntimeConfig:
    """Tests for RuntimeConfig and related functions."""

    def test_get_default_configs(self):
        """Test that get_default_configs returns expected runtime configuration."""
        configs = get_default_configs()

        assert configs == {"max_iterations": 4}
        assert isinstance(configs, dict)
        assert configs["max_iterations"] == 4

    def test_runtime_config_typed_dict(self):
        """Test that RuntimeConfig is a TypedDict with expected structure."""
        from src.config.configs import RuntimeConfig

        # This is a type check, but we can verify the structure exists
        config: RuntimeConfig = {"max_iterations": 4}
        assert config["max_iterations"] == 4

        # Should not allow missing keys
        with pytest.raises(KeyError):
            _ = config["invalid_key"]  # type: ignore
