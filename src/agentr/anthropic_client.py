"""Anthropic LLM client implementation."""

from typing import Any, AsyncIterator, Iterator

from anthropic import Anthropic, AsyncAnthropic

from agentr.llm_client import LLMClient


class AnthropicClient(LLMClient):
    """Anthropic LLM client implementation."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5-20250929", **kwargs):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model to use for completions.
            **kwargs: Additional parameters for Anthropic client initialization.
        """
        self.model = model
        self.client = Anthropic(api_key=api_key, **kwargs)
        self.async_client = AsyncAnthropic(api_key=api_key, **kwargs)

    def chat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Synchronous chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM (e.g., max_tokens, temperature).

        Returns:
            Response dictionary from the LLM.
        """
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", 1024)

        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )

        return response.model_dump()

    async def achat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Asynchronous chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM (e.g., max_tokens, temperature).

        Returns:
            Response dictionary from the LLM.
        """
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", 1024)

        response = await self.async_client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )

        return response.model_dump()

    def stream(self, messages: list[dict[str, Any]], **kwargs) -> Iterator[dict[str, Any]]:
        """
        Synchronous streaming chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM (e.g., max_tokens, temperature).

        Yields:
            Response chunks from the LLM.
        """
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", 1024)

        with self.client.messages.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        ) as stream:
            for chunk in stream:
                yield chunk.model_dump()

    async def astream(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronous streaming chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM (e.g., max_tokens, temperature).

        Yields:
            Response chunks from the LLM.
        """
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", 1024)

        async with self.async_client.messages.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        ) as stream:
            async for chunk in stream:
                yield chunk.model_dump()
