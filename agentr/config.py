from __future__ import annotations

import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class EnvConfig(BaseSettings):
    """Environment configuration for AgentR."""

    api_key: str
    api_url: str
    model_name: str
    tavily_api_key: str
    langfuse_base_url: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    log_level: str = "INFO"
    environment: str = "development"

    _instance: EnvConfig | None = None

    @classmethod
    def get_instance(cls) -> EnvConfig:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    model_config = {
        "env_file": ".env",
        "extra": "allow",
    }

    def model_post_init(self, __context):
        logger.info(f"Configuration loaded: model={self.model_name}, env={self.environment}")


def get_default_runtime_config() -> dict:
    """Get default runtime configuration."""
    return {"max_iterations": 4}
