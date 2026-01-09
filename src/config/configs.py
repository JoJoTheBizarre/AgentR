from pydantic_settings import BaseSettings


class ClientSettings(BaseSettings):
    api_key: str
    api_url: str
    model_name: str
