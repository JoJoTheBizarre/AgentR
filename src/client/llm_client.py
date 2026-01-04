"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Union, List

from .message_types import Message


class LLMClient(ABC):
    """Abstract base class for LLM client implementations."""

    @abstractmethod
    def chat(self, messages: List[Union[Message, dict[str, Any]]], **kwargs: Any) -> dict[str, Any]:
        """
        Synchronous chat completion.

        Args:
            messages: List of Message instances and/or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Response dictionary from the LLM.
        """
        pass

    @abstractmethod
    async def achat(self, messages: List[Union[Message, dict[str, Any]]], **kwargs: Any) -> dict[str, Any]:
        """
        Asynchronous chat completion.

        Args:
            messages: List of Message instances and/or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Response dictionary from the LLM.
        """
        pass

    @abstractmethod
    def stream(self, messages: List[Union[Message, dict[str, Any]]], **kwargs: Any) -> Iterator[dict[str, Any]]:
        """
        Synchronous streaming chat completion.

        Args:
            messages: List of Message instances and/or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM.
        """
        pass

    @abstractmethod
    async def astream(self, messages: List[Union[Message, dict[str, Any]]], **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronous streaming chat completion.

        Args:
            messages: List of Message instances and/or raw message dictionaries.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM.
        """
        pass
