import logging

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import EnvConfig
from src.exceptions import ClientInitializationError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    A wrapper around ChatOpenAI with optional structured tool support.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client_config: EnvConfig | None = None,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            model (str | None): The model name.
            api_key (str | None): The OpenAI API key.
            client_config (ClientSettings | None): Optional config object.

        Raises:
            ClientInitializationError: If necessary arguments are not provided.
        """
        if client_config:
            self.api_key = client_config.api_key
            self.model = client_config.model_name
            self.api_url = client_config.api_url
        elif api_key and model:
            self.api_key = api_key
            self.model = model
        else:
            raise ClientInitializationError(
                "You must provide either client_config or both api_key and model."
            )

        self.client = ChatOpenAI(
            model=self.model,
            api_key=SecretStr(self.api_key),
            base_url=self.api_url,
        )
        logger.info(f"OpenAI client initialized with model: {self.model}")

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Send messages to the ChatOpenAI client and get a response.

        Args:
            messages (List[BaseMessage]): List of messages to send.

        Returns:
            AIMessage: The response from the model.
        """
        logger.debug(f"Chat API call with {len(messages)} messages")
        try:
            response = self.client.invoke(input=messages)
            # Log token usage if available
            if (
                hasattr(response, "response_metadata")
                and "token_usage" in response.response_metadata
            ):
                tokens = response.response_metadata["token_usage"]
                logger.debug(
                    f"API call successful: {tokens.get('total_tokens', 'N/A')} tokens"
                )
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
        """
        Invoke the LLM with structured tools.

        Args:
            messages (list[BaseMessage]): List of messages to process.
            tools (list[StructuredTool]): Tools to bind to the LLM.
            parallel (bool, optional): Whether to run tools in parallel.
                Defaults to False.

        Returns:
            AIMessage: The response from the model with potential tool calls.
        """
        logger.debug(
            f"Structured output API call with {len(messages)} messages, {len(tools)} tools"
        )
        try:
            llm_with_tools = self.client.bind_tools(
                tools=tools, parallel_tool_calls=parallel
            )
            response = llm_with_tools.invoke(input=messages)

            # Log token usage if available
            if (
                hasattr(response, "response_metadata")
                and "token_usage" in response.response_metadata
            ):
                tokens = response.response_metadata["token_usage"]
                logger.debug(
                    f"Structured API call successful: {tokens.get('total_tokens', 'N/A')} tokens"
                )

            return response
        except Exception as e:
            logger.error(f"Structured output API call failed: {type(e).__name__}")
            raise
