"""OpenAI LLM client implementation."""

import os
from typing import Any, AsyncIterator, Dict, Iterator, cast, List, Optional

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)

from .base import LLMClient
from ..core.messages import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    ToolCall,
    FunctionCall,
    Role,
)


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model: Optional[str] = model or os.getenv("OPENAI_MODEL")

        self.client: OpenAI = OpenAI(api_key=api_key, **kwargs)
        self.async_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, **kwargs)

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _get_model(self, model_override: Optional[str]) -> str:
        model: Optional[str] = model_override or self.model
        if not model:
            raise ValueError(
                "No OpenAI model specified. "
                "Pass `model` in the call, pass `model` to OpenAIClient, "
                "or set OPENAI_MODEL env variable."
            )
        return model

    # ------------------------------------------------------------------
    # OpenAI → Custom message
    # ------------------------------------------------------------------

    @staticmethod
    def _customize_response(response: ChatCompletion) -> AssistantMessage:
        if not response.choices:
            raise ValueError("No choices in response")

        msg = response.choices[0].message

        tool_calls: Optional[List[ToolCall]] = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type=tc.type,  # type: ignore
                        function=FunctionCall(
                            name=tc.function.name,  # type: ignore
                            arguments=tc.function.arguments,  # type: ignore
                        ),
                    )
                )

        return AssistantMessage(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # Custom → OpenAI messages
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_messages(
        system_message: SystemMessage,
        messages: List[Message],
    ) -> List[ChatCompletionMessageParam]:
        """
        Convert custom message types into OpenAI-compatible chat messages.
        """

        openai_messages: List[ChatCompletionMessageParam] = []

        # ---- system message (must be first)
        openai_messages.append(
            {
                "role": Role.SYSTEM.value,
                "content": system_message.content,
            }
        )

        for msg in messages:
            # -------------------------
            # User message
            # -------------------------
            if isinstance(msg, UserMessage):
                openai_messages.append(
                    {
                        "role": Role.USER.value,
                        "content": msg.content,
                    }
                )

            # -------------------------
            # Assistant message
            # -------------------------
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

                openai_messages.append(
                    cast(ChatCompletionMessageParam, payload)
                )

            # -------------------------
            # Tool result message
            # -------------------------
            elif isinstance(msg, ToolResultMessage):
                openai_messages.append(
                    {
                        "role": Role.TOOL.value,
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )

            else:
                # Should be unreachable if Message union is exhaustive
                raise TypeError(f"Unsupported message type: {type(msg)}")

        return openai_messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_message: SystemMessage,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        model: str = self._get_model(kwargs.pop("model", None))

        openai_messages = self._to_openai_messages(system_message, messages)

        response: ChatCompletion = self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=tools,
            **kwargs,
        )

        return self._customize_response(response)

    async def achat(
        self,
        system_message: SystemMessage,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        raise NotImplementedError("Async chat not implemented for OpenAIClient")

    def stream(
        self,
        system_message: SystemMessage,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError("Streaming not implemented for OpenAIClient")

    async def astream(
        self,
        system_message: SystemMessage,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError("Async streaming not implemented for OpenAIClient")
