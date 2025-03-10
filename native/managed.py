from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
from typing import Generic, TypeVar, Any, Iterator, Self, AsyncIterator

from native.types import LoopProtocol

V = TypeVar("V")
U = TypeVar("U")


class ManagedValue(ABC, Generic[V]):
    def __init__(self, loop: LoopProtocol) -> None:
        self.loop = loop

    @classmethod
    @contextmanager
    def enter(cls, loop: LoopProtocol, **kwargs: Any) -> Iterator[Self]:
        try:
            value = cls(loop, **kwargs)
            yield value
        finally:
            # because managed value and Pregel have reference to each other
            # let's make sure to break the reference on exit
            try:
                del value
            except UnboundLocalError:
                pass

    @classmethod
    @asynccontextmanager
    async def aenter(cls, loop: LoopProtocol, **kwargs: Any) -> AsyncIterator[Self]:
        try:
            value = cls(loop, **kwargs)
            yield value
        finally:
            # because managed value and Pregel have reference to each other
            # let's make sure to break the reference on exit
            try:
                del value
            except UnboundLocalError:
                pass

    @abstractmethod
    def __call__(self) -> V:
        ...
