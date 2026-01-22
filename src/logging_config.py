"""
Centralized logging configuration for AgentR.

Configure logging based on environment settings from EnvConfig.
"""

import logging
import sys

from src.config import EnvConfig


def setup_logging(config: EnvConfig | None = None) -> None:
    """
    Configure logging based on environment configuration.

    Args:
        config: EnvConfig instance. If None, loads default configuration.

    Behavior:
        - Development environment: Human-readable logs to stdout
        - Production environment: Human-readable logs to stdout (can extend to JSON/file later)
        - Sets log level from config.log_level (defaults to INFO)
        - Suppresses verbose third-party library logs
    """
    if config is None:
        config = EnvConfig()  # type: ignore

    # Map string log level to logging constant
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    # Clear any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure basic logging
    if config.environment.lower() == "production":
        # Production: simpler format, potentially to file in future
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        # Development: more verbose with line numbers
        format_string = (
            "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
        )

    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Log the logging configuration
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Logging configured: level={config.log_level}, env={config.environment}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper configuration.

    This is a convenience wrapper that ensures logging is configured.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
