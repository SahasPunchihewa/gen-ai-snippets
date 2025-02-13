from typing import Optional, Callable, Self, cast, Literal, Any

from langchain_core.runnables import RunnableConfig

StreamMode = Literal["values", "updates", "debug", "messages", "custom"]
"""How the stream method should emit outputs.

- 'values': Emit all values of the state for each step.
- 'updates': Emit only the node name(s) and updates
    that were returned by the node(s) **after** each step.
- 'debug': Emit debug events for each step.
- 'messages': Emit LLM messages token-by-token.
- 'custom': Emit custom output `write: StreamWriter` kwarg of each node.
"""

StreamChunk = tuple[tuple[str, ...], str, Any]


class StreamProtocol:
    __slots__ = ("modes", "__call__")

    modes: set[StreamMode]

    __call__: Callable[[Self, StreamChunk], None]

    def __init__(
            self,
            __call__: Callable[[StreamChunk], None],
            modes: set[StreamMode],
    ) -> None:
        self.__call__ = cast(Callable[[Self, StreamChunk], None], __call__)
        self.modes = modes


class LoopProtocol:
    config: RunnableConfig
    store: Optional["BaseStore"]
    stream: Optional[StreamProtocol]
    step: int
    stop: int

    def __init__(
            self,
            *,
            step: int,
            stop: int,
            config: RunnableConfig,
            store: Optional["BaseStore"] = None,
            stream: Optional[StreamProtocol] = None,
    ) -> None:
        self.stream = stream
        self.config = config
        self.store = store
        self.step = step
        self.stop = stop
