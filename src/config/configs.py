import logging
from typing import Any

from pydantic_settings import BaseSettings
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class EnvConfig(BaseSettings):
    api_key: str
    api_url: str
    model_name: str
    tavily_api_key: str
    langfuse_base_url: str
    langfuse_public_key: str
    langfuse_secret_key: str

    class Config:
        env_file = ".env.dev"
        extra = "allow"

    def model_post_init(self, __context: Any) -> None:
        """Log configuration loading and validate optional settings."""
        super().model_post_init(__context)
        logger.info(f"Configuration loaded from {self.Config.env_file}")
        logger.info(f"Model: {self.model_name}, API URL: {self.api_url}")

        # Log warnings for potentially missing optional configurations
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set - research queries will fail")
        if not self.langfuse_public_key or not self.langfuse_secret_key:
            logger.warning("Langfuse keys not set - tracing will be disabled")


class RuntimeConfig(TypedDict):
    max_iterations: int


def get_default_configs() -> dict[str, Any]:
    return dict(RuntimeConfig(max_iterations=4))
