from pydantic_settings import BaseSettings


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
