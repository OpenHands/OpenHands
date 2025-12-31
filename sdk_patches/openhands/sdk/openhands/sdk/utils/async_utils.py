"""Async utilities for OpenHands SDK.

This module provides utilities for working with async callbacks in the context
of synchronous conversation handling.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from openhands.sdk.event.base import Event


AsyncConversationCallback = Callable[[Event], Coroutine[Any, Any, None]]


class AsyncCallbackWrapper:
    """Wrapper that executes async callbacks in a different thread's event loop.

    This class implements the ConversationCallbackType interface (synchronous)
    but internally executes an async callback in an event loop running in a
    different thread. This allows async callbacks to be used in synchronous
    conversation contexts.
    """

    async_callback: AsyncConversationCallback
    loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        async_callback: AsyncConversationCallback,
        loop: asyncio.AbstractEventLoop,
    ):
        self.async_callback = async_callback
        self.loop = loop

    def __call__(self, event: Event):
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.async_callback(event), self.loop)
