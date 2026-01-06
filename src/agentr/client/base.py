"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from agentr.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
)


class LLMClient(ABC):
    """Abstract base class for LLM client implementations."""

    @abstractmethod
    def chat(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> AssistantMessage:
        """
        Synchronous chat completion.

        Args:
            system_message: System instructions for the LLM.
            messages: List of conversation messages.
            tools: Optional list of tool definitions available to the LLM.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Assistant response message, potentially containing tool calls.
        """
        pass

    @abstractmethod
    async def achat(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> AssistantMessage:
        """
        Asynchronous chat completion.

        Args:
            system_message: System instructions for the LLM.
            messages: List of conversation messages.
            tools: Optional list of tool definitions available to the LLM.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Assistant response message, potentially containing tool calls.
        """
        pass

    @abstractmethod
    def stream(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> Iterator[dict[str, object]]:
        """
        Synchronous streaming chat completion.

        Args:
            system_message: System instructions for the LLM.
            messages: List of conversation messages.
            tools: Optional list of tool definitions available to the LLM.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM (provider-specific format).
        """
        pass

    @abstractmethod
    async def astream(
        self,
        system_message: SystemMessage,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, object]]:
        """
        Asynchronous streaming chat completion.

        Args:
            system_message: System instructions for the LLM.
            messages: List of conversation messages.
            tools: Optional list of tool definitions available to the LLM.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM (provider-specific format).
        """
        pass
