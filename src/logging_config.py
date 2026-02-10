import logging
import sys

from src.config import EnvConfig


def setup_logging(config: EnvConfig | None = None) -> None:
    """Configure logging based on environment configuration."""
    if config is None:
        config = EnvConfig()  # type: ignore

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if config.environment.lower() == "production":
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        format_string = (
            "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
        )

    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

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
