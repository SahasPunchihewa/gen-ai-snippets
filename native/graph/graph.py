import uuid
import warnings
from functools import partial
from typing import Callable, Union, Optional, Any, cast, Literal, Sequence

from langchain_core.messages import MessageLikeRepresentation, message_chunk_to_message, convert_to_messages, BaseMessageChunk, RemoveMessage, BaseMessage

from native.graph.state import StateGraph

Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]


def _format_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    try:
        from langchain_core.messages import convert_to_openai_messages
    except ImportError:
        msg = (
            "Must have langchain-core>=0.3.11 installed to use automatic message "
            "formatting (format='langchain-openai'). Please update your langchain-core "
            "version or remove the 'format' flag. Returning un-formatted "
            "messages."
        )
        warnings.warn(msg)
        return list(messages)
    else:
        return convert_to_messages(convert_to_openai_messages(messages))


def _add_messages_wrapper(func: Callable) -> Callable[[Messages, Messages], Messages]:
    def _add_messages(
            left: Optional[Messages] = None, right: Optional[Messages] = None, **kwargs: Any
    ) -> Union[Messages, Callable[[Messages, Messages], Messages]]:
        if left is not None and right is not None:
            return func(left, right, **kwargs)
        elif left is not None or right is not None:
            msg = (
                f"Must specify non-null arguments for both 'left' and 'right'. Only "
                f"received: '{'left' if left else 'right'}'."
            )
            raise ValueError(msg)
        else:
            return partial(func, **kwargs)

    _add_messages.__doc__ = func.__doc__
    return cast(Callable[[Messages, Messages], Messages], _add_messages)


@_add_messages_wrapper
def add_messages(
        left: Messages,
        right: Messages,
        *,
        format: Optional[Literal["langchain-openai"]] = None,
) -> Messages:
    # coerce to list
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]
    # coerce to message
    left = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(left)
    ]
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]
    # assign missing ids
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for m in right:
        if m.id is None:
            m.id = str(uuid.uuid4())
    # merge
    left_idx_by_id = {m.id: i for i, m in enumerate(left)}
    merged = left.copy()
    ids_to_remove = set()
    for m in right:
        if (existing_idx := left_idx_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )

            merged.append(m)
    merged = [m for m in merged if m.id not in ids_to_remove]

    if format == "langchain-openai":
        merged = _format_messages(merged)
    elif format:
        msg = f"Unrecognized {format=}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)
    else:
        pass

    return merged


class MessageGraph(StateGraph):

    def __init__(self) -> None:
        super().__init__(Annotated[list[AnyMessage], add_messages])  # type: ignore[arg-type]
