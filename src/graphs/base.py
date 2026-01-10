from abc import ABC, abstractmethod
from typing import Any


class BaseNode(ABC):
    @abstractmethod
    def _execute(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._execute(*args, **kwargs)
