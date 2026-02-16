import logging
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import EnvConfig
from .exceptions import ClientInitializationError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper around ChatOpenAI with structured tool support."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        client_config: EnvConfig | None = None,
    ):
        """Initialize the OpenAI client."""
        if client_config:
            self.api_key = client_config.api_key
            self.model = client_config.model_name
            self.api_url = client_config.api_url
        elif api_key and model:
            self.api_key = api_key
            self.model = model
            self.api_url = api_url or "https://api.openai.com/v1"
        else:
            raise ClientInitializationError(
                "Provide either client_config or both api_key and model"
            )

        self.client = ChatOpenAI(
            model=self.model,
            api_key=SecretStr(self.api_key),
            base_url=self.api_url,
        )
        logger.info(f"OpenAI client initialized: model={self.model}")

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        """Send messages and get response."""
        logger.debug(f"Chat API call with {len(messages)} messages")
        try:
            response = self.client.invoke(input=messages)
            if (
                hasattr(response, "response_metadata")
                and "token_usage" in response.response_metadata
            ):
                tokens = response.response_metadata["token_usage"]
                logger.debug(f"API call: {tokens.get('total_tokens', 'N/A')} tokens")
            return response
        except Exception as e:
            logger.error(f"Chat API call failed: {type(e).__name__}")
            raise

    def with_structured_output(
        self,
        messages: list[BaseMessage],
        tools: list[StructuredTool],
        parallel: bool = False,
    ) -> AIMessage:
        """Invoke LLM with structured tools."""
        logger.debug(f"Structured output: {len(messages)} messages, {len(tools)} tools")
        try:
            llm_with_tools = self.client.bind_tools(tools=tools, parallel_tool_calls=parallel)
            return llm_with_tools.invoke(input=messages)
        except Exception as e:
            logger.error(f"Structured output failed: {type(e).__name__}")
            raise
