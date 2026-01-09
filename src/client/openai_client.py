from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from config import ClientSettings


class OpenAIClient:
    """
    A wrapper around ChatOpenAI with optional structured tool support.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client_config: ClientSettings | None = None,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            model (str | None): The model name.
            api_key (str | None): The OpenAI API key.
            client_config (ClientSettings | None): Optional config object.

        Raises:
            ValueError: If necessary arguments are not provided.
        """
        if client_config:
            self.api_key = client_config.api_key
            self.model = client_config.model_name
            self.api_url = client_config.api_url
        elif api_key and model:
            self.api_key = api_key
            self.model = model
        else:
            raise ValueError(
                "You must provide either client_config or both api_key and model."
            )

        self.client = ChatOpenAI(model=self.model, api_key=str(self.api_key))  # pyright: ignore[reportArgumentType]

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Send messages to the ChatOpenAI client and get a response.

        Args:
            messages (List[BaseMessage]): List of messages to send.

        Returns:
            AIMessage: The response from the model.
        """
        response = self.client.invoke(input=messages)
        return response

    def with_structured_output(
        self,
        messages: list[BaseMessage],
        tools: list[StructuredTool],
        parallel: bool = False,
        choice: str = "first",
    ) -> AIMessage | ToolCall | list[ToolCall]:
        """
        Invoke the LLM with structured tools.

        Args:
            messages (List[BaseMessage]): List of messages to process.
            tools (List[StructuredTool]): Tools to bind to the LLM.
            parallel (bool, optional): Whether to run tools in parallel. Defaults to False.
            choice (str, optional): 'first' to return only the first tool call, 'all' to return all.

        Returns:
            AIMessage | list[ToolCall]: AIMessage if no tools are called, else ToolCall(s).
        """  # noqa: E501
        llm_with_tools = self.client.bind_tools(tools=tools, parallel=parallel)
        response = llm_with_tools.invoke(input=messages, choice=choice)

        if response.tool_calls:
            if choice == "first":
                return response.tool_calls[0]
            else:
                return response.tool_calls
        return response
