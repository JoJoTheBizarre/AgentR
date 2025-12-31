"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator


class LLMClient(ABC):
    """Abstract base class for LLM client implementations."""

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Synchronous chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Response dictionary from the LLM.
        """
        pass

    @abstractmethod
    async def achat(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Asynchronous chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM.

        Returns:
            Response dictionary from the LLM.
        """
        pass

    @abstractmethod
    def stream(self, messages: list[dict[str, Any]], **kwargs) -> Iterator[dict[str, Any]]:
        """
        Synchronous streaming chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM.
        """
        pass

    @abstractmethod
    async def astream(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronous streaming chat completion.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional parameters for the LLM.

        Yields:
            Response chunks from the LLM.
        """
        pass
