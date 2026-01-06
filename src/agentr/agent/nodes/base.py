import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar

from langchain_core.runnables import RunnableConfig

S = TypeVar("S")
T = TypeVar("T")


class BaseNode(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    def execute(self, state: S, config: RunnableConfig) -> T:
        """Sync execution wrapper"""
        return asyncio.run(self._execute(state, config))

    async def aexecute(self, state: S, config: RunnableConfig) -> T:
        """Async execution wrapper"""
        return await self._execute(state, config)

    @abstractmethod
    async def _execute(self, state: S, config: RunnableConfig) -> T:
        """Override this method with your node logic"""
        pass

    def __call__(self, state: S, config: RunnableConfig) -> T:
        return self.execute(state, config)
