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

    class Config:
        env_file = ".env"
        extra = "allow"

    def model_post_init(self, __context):
        """Post-initialization logging and validation."""
        logger.info(f"Configuration loaded: model={self.model_name}, env={self.environment}")

        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set - web search will not work")

        if not self.langfuse_public_key or not self.langfuse_secret_key:
            logger.warning("Langfuse keys not set - tracing disabled")


def get_default_runtime_config() -> dict:
    """Get default runtime configuration."""
    return {"max_iterations": 4}
