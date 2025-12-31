from collections.abc import Sequence

from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.llm import LLM


def get_total_token_count(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
) -> int:
    """Calculate the total token count for a list of LLM convertible events.

    This function converts the events to LLM messages and uses the provided LLM
    to count the total number of tokens. This is useful for understanding how many
    tokens a sequence of events will consume in the context window.

    Args:
        events: List of LLM convertible events to count tokens for
        llm: The LLM instance to use for token counting (uses the litellm's token
            counting utilities)

    Returns:
        Total token count for all events converted to messages

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Hello, how are you?", source="user"),
        ...     MessageEvent.from_text("I'm doing great!", source="agent"),
        ... ]
        >>> token_count = get_total_token_count(events, llm)
        >>> print(f"Total tokens: {token_count}")
    """
    messages = LLMConvertibleEvent.events_to_messages(list(events))
    return llm.get_token_count(messages)


def get_shortest_prefix_above_token_count(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
    token_count: int,
) -> int:
    """Find the length of the shortest prefix whose token count exceeds the target.

    This function performs a binary search to efficiently find the shortest prefix
    of events that, when converted to messages, has a total token count greater than
    the specified target token count.

    Args:
        events: List of LLM convertible events to search through
        llm: The LLM instance to use for token counting (uses the model's tokenizer)
        token_count: The target token count threshold

    Returns:
        The length of the shortest prefix that exceeds the token count.
        Returns 0 if no events are provided.
        Returns len(events) if all events combined don't exceed the token count.

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Hi", source="user"),
        ...     MessageEvent.from_text("Hello", source="agent"),
        ...     MessageEvent.from_text("How are you?", source="user"),
        ...     MessageEvent.from_text("Great!", source="agent"),
        ... ]
        >>> prefix_len = get_shortest_prefix_above_token_count(events, llm, 20)
        >>> # prefix_len might be 2 if first 2 events exceed 20 tokens
    """
    if not events:
        return 0

    # Check if all events combined don't exceed the token count
    total_tokens = get_total_token_count(events, llm)
    if total_tokens <= token_count:
        return len(events)

    # Binary search for the shortest prefix
    left, right = 1, len(events)

    while left < right:
        mid = (left + right) // 2
        prefix_tokens = get_total_token_count(events[:mid], llm)

        if prefix_tokens > token_count:
            # This prefix exceeds the count, try to find a shorter one
            right = mid
        else:
            # This prefix doesn't exceed, we need a longer one
            left = mid + 1

    return left


def get_suffix_length_for_token_reduction(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
    token_reduction: int,
) -> int:
    """Find how many suffix events can be kept while reducing tokens by target amount.

    This function determines the maximum number of events from the end of the list
    that can be retained while ensuring the total token count is reduced by at least
    the specified amount. It uses the get_shortest_prefix_above_token_count function
    to find the prefix that must be removed.

    Args:
        events: List of LLM convertible events
        llm: The LLM instance to use for token counting (uses the model's tokenizer)
        token_reduction: The minimum number of tokens to reduce by

    Returns:
        The number of events from the end that can be kept (suffix length).

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Event 1", source="user"),
        ...     MessageEvent.from_text("Event 2", source="agent"),
        ...     MessageEvent.from_text("Event 3", source="user"),
        ...     MessageEvent.from_text("Event 4", source="agent"),
        ... ]
        >>> # Suppose total is 100 tokens, and we want to reduce by 40 tokens
        >>> suffix_len = get_suffix_length_for_token_reduction(events, llm, 40)
        >>> # suffix_len tells us how many events from the end we can keep
        >>> # If first 2 events = 45 tokens, suffix_len = 2 (keep last 2 events)
    """
    if not events:
        return 0

    if token_reduction <= 0:
        return len(events)

    # Find the shortest prefix that exceeds the token reduction target
    prefix_length = get_shortest_prefix_above_token_count(events, llm, token_reduction)

    # The suffix length is what remains after removing the prefix
    suffix_length = len(events) - prefix_length

    return suffix_length
