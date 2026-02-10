import logging
import sys
from unittest.mock import patch

from src.config import EnvConfig
from src.logging_config import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_development_environment_format(self, mock_env_vars):
        """Test logging format in development environment."""
        with patch.dict(
            "os.environ", {"ENVIRONMENT": "development", "LOG_LEVEL": "DEBUG"}
        ):
            config = EnvConfig()  # type: ignore
            setup_logging(config)

            root_logger = logging.getLogger()
            handler = root_logger.handlers[0]
            formatter = handler.formatter

            # Check format string includes line numbers for development
            format_string = formatter._fmt
            assert "%(lineno)d" in format_string
            assert "%(name)s" in format_string
            assert "%(levelname)s" in format_string

    def test_production_environment_format(self, mock_env_vars):
        """Test logging format in production environment."""
        with patch.dict(
            "os.environ", {"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"}
        ):
            config = EnvConfig()  # type: ignore
            setup_logging(config)

            root_logger = logging.getLogger()
            handler = root_logger.handlers[0]
            formatter = handler.formatter

            # Check format string doesn't include line numbers for production
            format_string = formatter._fmt
            assert "%(lineno)d" not in format_string
            assert "%(name)s" in format_string
            assert "%(levelname)s" in format_string

    def test_log_level_resolution(self, mock_env_vars):
        """Test that log level is correctly resolved from string."""
        test_cases = [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]

        for level_str, expected_level in test_cases:
            with patch.dict("os.environ", {"LOG_LEVEL": level_str}):
                config = EnvConfig()  # type: ignore
                setup_logging(config)

                root_logger = logging.getLogger()
                assert root_logger.level == expected_level

    def test_invalid_log_level_defaults_to_info(self, mock_env_vars):
        """Test that invalid log level defaults to INFO."""
        with patch.dict("os.environ", {"LOG_LEVEL": "INVALID_LEVEL"}):
            config = EnvConfig()  # type: ignore
            setup_logging(config)

            root_logger = logging.getLogger()
            assert root_logger.level == logging.INFO

    def test_clears_existing_handlers(self, mock_env_vars):
        """Test that setup_logging clears existing handlers."""
        config = EnvConfig()  # type: ignore

        root_logger = logging.getLogger()
        dummy_handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(dummy_handler)

        setup_logging(config)

        assert dummy_handler not in root_logger.handlers
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    def test_force_parameter_overrides_config(self, mock_env_vars):
        """Test that force=True overrides existing configuration."""
        config = EnvConfig()  # type: ignore

        setup_logging(config)
        first_handler = logging.getLogger().handlers[0]

        setup_logging(config)
        second_handler = logging.getLogger().handlers[0]

        assert first_handler is not second_handler
        assert len(logging.getLogger().handlers) == 1

    def test_default_config_when_none_provided(self, mock_env_vars):
        """Test that setup_logging works when no config is provided."""
        setup_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

        config = EnvConfig()  # type: ignore
        assert config.log_level == "DEBUG"
        assert root_logger.level == logging.DEBUG


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self, mock_env_vars):
        """Test that get_logger returns a configured logger."""
        setup_logging()
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
        assert logger.getEffectiveLevel() == logging.DEBUG  # From mock_env_vars

    def test_get_logger_without_setup(self, mock_env_vars):
        """Test that get_logger works even without explicit setup_logging call."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        logger = get_logger("another.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_inherits_configuration(self, mock_env_vars):
        """Test that logger inherits root logger configuration."""
        with patch.dict("os.environ", {"LOG_LEVEL": "WARNING"}):
            setup_logging()
            logger = get_logger("test.module")

            assert logger.getEffectiveLevel() == logging.WARNING
