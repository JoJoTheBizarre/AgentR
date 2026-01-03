"""OpenAI LLM client implementation."""

import os
from typing import Any, AsyncIterator, Iterator, cast, Union, List

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agentr.llm_client import LLMClient
from agentr.message_types import Message, messages_to_dicts

DEFAULT_MAX_TOKENS = 1024


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation."""

    def __init__(self, api_key: str | None = None, model: str | None = None, **kwargs):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use. If None, uses OPENAI_MODEL env var.
            **kwargs: Additional parameters for OpenAI client.
        """
        # Resolve once, but allow None (validated at call time)
        self.model = model or os.getenv("OPENAI_MODEL")

        self.default_max_tokens = int(
            os.getenv("OPENAI_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
        )

        self.client = OpenAI(api_key=api_key, **kwargs)
        self.async_client = AsyncOpenAI(api_key=api_key, **kwargs)

    def _get_model(self, model_override: str | None) -> str:
        """
        Return the model to use, validating at call time.
        """
        model = model_override or self.model
        if not model:
            raise ValueError(
                "No OpenAI model specified. "
                "Pass `model` in the call, pass `model` to OpenAIClient, "
                "or set OPENAI_MODEL env variable."
            )
        return model

    def _normalize_messages(self, messages: Union[List[Message], List[dict[str, Any]]]) -> List[dict[str, Any]]:
        """
        Convert messages to list of dicts compatible with OpenAI API.

        Args:
            messages: List of Message instances or raw message dicts.

        Returns:
            List of message dictionaries.
        """
        if not messages:
            return []

        normalized = []
        for msg in messages:
            if hasattr(msg, 'to_dict'):
                # Message instance
                normalized.append(msg.to_dict())
            else:
                # Assume it's already a dict
                normalized.append(msg)  # type: ignore

        return normalized

    def chat(self, messages: Union[List[Message], List[dict[str, Any]]], **kwargs) -> dict[str, Any]:
        """
        Synchronous chat completion.

        Args:
            messages: List of Message instances or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Response dictionary from the LLM.
        """
        model = self._get_model(kwargs.pop("model", None))
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)

        normalized_messages = self._normalize_messages(messages)
        cast_messages = cast(list[ChatCompletionMessageParam], normalized_messages)
        response = self.client.chat.completions.create(
            model=model,
            messages=cast_messages,
            max_tokens=max_tokens,
            **kwargs,
        )

        return response.model_dump()

    async def achat(self, messages: Union[List[Message], List[dict[str, Any]]], **kwargs) -> dict[str, Any]:
        """
        Asynchronous chat completion.

        Args:
            messages: List of Message instances or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Response dictionary from the LLM.
        """
        model = self._get_model(kwargs.pop("model", None))
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)

        normalized_messages = self._normalize_messages(messages)
        cast_messages = cast(list[ChatCompletionMessageParam], normalized_messages)
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=cast_messages,
            max_tokens=max_tokens,
            **kwargs,
        )

        return response.model_dump()

    def stream(
        self, messages: Union[List[Message], List[dict[str, Any]]], **kwargs
    ) -> Iterator[dict[str, Any]]:
        """
        Synchronous streaming chat completion.

        Args:
            messages: List of Message instances or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM.
        """
        model = self._get_model(kwargs.pop("model", None))
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)

        normalized_messages = self._normalize_messages(messages)
        cast_messages = cast(list[ChatCompletionMessageParam], normalized_messages)
        stream = self.client.chat.completions.create(
            model=model,
            messages=cast_messages,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            yield chunk.model_dump()

    async def astream(
        self, messages: Union[List[Message], List[dict[str, Any]]], **kwargs
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronous streaming chat completion.

        Args:
            messages: List of Message instances or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM.
        """
        model = self._get_model(kwargs.pop("model", None))
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)

        normalized_messages = self._normalize_messages(messages)
        cast_messages = cast(list[ChatCompletionMessageParam], normalized_messages)
        stream = await self.async_client.chat.completions.create(
            model=model,
            messages=cast_messages,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            yield chunk.model_dump()
