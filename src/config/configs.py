from pydantic_settings import BaseSettings


class ClientSettings(BaseSettings):
    api_key: str
    api_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    tavily_api_key: str = ""

    class Config:
        env_file = (".env.dev", ".env")  # Try .env.dev first, then .env
        extra = "allow"
