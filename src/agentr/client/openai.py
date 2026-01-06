"""OpenAI LLM client implementation."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from agentr.client.base import LLMClient
from agentr.core.messages import (
    AssistantMessage,
    FunctionCall,
    Message,
    Role,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation."""

    def __init__(self, api_key: str | None = None, **kwargs: object) -> None:
        """
        Initialize OpenAI client.

        Args:
            api_key: Optional OpenAI API key.
            kwargs: Additional keyword arguments for OpenAI client.
        """
        self.client: OpenAI = OpenAI(api_key=api_key, **kwargs)
        self.async_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, **kwargs)

    @staticmethod
    def _customize_response(response: ChatCompletion) -> AssistantMessage:
        """
        Convert OpenAI chat response into an AssistantMessage.

        Args:
            response: OpenAI ChatCompletion response.

        Returns:
            AssistantMessage containing the assistant's content and any tool calls.
        """
        if not response.choices:
            raise ValueError("No choices in response")

        msg = response.choices[0].message

        tool_calls: list[ToolCall] | None = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,  # type: ignore
                    function=FunctionCall(
                        name=tc.function.name,  # type: ignore
                        arguments=tc.function.arguments,  # type: ignore
                    ),
                )
                for tc in msg.tool_calls
            ]

        return AssistantMessage(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _to_openai_messages(
        system_message: SystemMessage,
        messages: list[Message],
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert custom message types into OpenAI-compatible chat messages.

        Args:
            system_message: The system message to start the conversation.
            messages: List of Message objects (user, assistant, or tool results).

        Returns:
            A list of ChatCompletionMessageParam for OpenAI.
        """
        openai_messages: list[ChatCompletionMessageParam] = []

        # Add system message first
        openai_messages.append(
            {
                "role": Role.SYSTEM.value,
                "content": system_message.content,
            }
        )

        for msg in messages:
            if isinstance(msg, UserMessage):
                openai_messages.append(
                    {
                        "role": Role.USER.value,
                        "content": msg.content,
                    }
                )

            elif isinstance(msg, AssistantMessage):
                payload: dict[str, Any] = {
                    "role": Role.ASSISTANT.value,
                    "content": msg.content,
                }

                if msg.tool_calls:
                    payload["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                openai_messages.append(cast(ChatCompletionMessageParam, payload))

            elif isinstance(msg, ToolResultMessage):
                openai_messages.append(
                    {
                        "role": Role.TOOL.value,
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )

            else:
                raise TypeError(f"Unsupported message type: {type(msg)}")

        return openai_messages

    def chat(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> AssistantMessage:
        """
        Perform a synchronous chat with the OpenAI model.

        Args:
            system_message: SystemMessage to start the conversation.
            messages: List of Message objects.
            tools: Optional list of tools/functions for the model.
            kwargs: Additional keyword arguments for the OpenAI API.

        Returns:
            AssistantMessage with the model's response.
        """
        model: str = kwargs.pop("model", None)
        if model is None:
            raise ValueError("Model must be specified for OpenAIClient chat")

        openai_messages = self._to_openai_messages(system_message, messages)
        if tools:
            kwargs["tools"] = tools

        response: ChatCompletion = self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            **kwargs,
        )

        return self._customize_response(response)

    async def achat(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> AssistantMessage:
        """
        Async chat not implemented for OpenAIClient.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Async chat not implemented for OpenAIClient")

    def stream(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> Iterator[dict[str, Any]]:
        """
        Streaming chat not implemented for OpenAIClient.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Streaming not implemented for OpenAIClient")

    async def astream(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Async streaming chat not implemented for OpenAIClient.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Async streaming not implemented for OpenAIClient")
