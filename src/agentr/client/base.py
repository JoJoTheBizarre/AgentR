"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..core.messages import (
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
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
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
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
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
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
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
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
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