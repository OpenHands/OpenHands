"""This module monitors the app for shutdown signals. This exists because the atexit module
does not play nocely with stareltte / uvicorn shutdown signals.
"""

import signal
import threading
import time
from types import FrameType
from typing import Callable
from uuid import UUID, uuid4

from uvicorn.server import HANDLED_SIGNALS

from openhands.app_server.utils.logger import openhands_logger as logger

_should_exit = None
_shutdown_listeners: dict[UUID, Callable] = {}


def _register_signal_handler(sig: signal.Signals) -> None:
    original_handler = None

    def handler(sig_: int, frame: FrameType | None) -> None:
        logger.debug(f'shutdown_signal:{sig_}')
        global _should_exit
        if not _should_exit:
            _should_exit = True
            listeners = list(_shutdown_listeners.values())
            for callable in listeners:
                try:
                    callable()
                except Exception:
                    logger.exception('Error calling shutdown listener')
            if original_handler:
                original_handler(sig_, frame)  # type: ignore[unreachable]

    original_handler = signal.signal(sig, handler)


def _register_signal_handlers() -> None:
    global _should_exit
    if _should_exit is not None:
        return
    _should_exit = False

    logger.debug('_register_signal_handlers')

    # Check if we're in the main thread of the main interpreter
    if threading.current_thread() is threading.main_thread():
        logger.debug('_register_signal_handlers:main_thread')
        for sig in HANDLED_SIGNALS:
            _register_signal_handler(sig)
    else:
        logger.debug('_register_signal_handlers:not_main_thread')


def should_exit() -> bool:
    _register_signal_handlers()
    return bool(_should_exit)


def should_continue() -> bool:
    _register_signal_handlers()
    return not _should_exit


def sleep_if_should_continue(timeout: float) -> None:
    if timeout <= 1:
        time.sleep(timeout)
        return
    start_time = time.time()
    while (time.time() - start_time) < timeout and should_continue():
        time.sleep(1)


def add_shutdown_listener(callable: Callable) -> UUID:
    id_ = uuid4()
    _shutdown_listeners[id_] = callable
    return id_


def remove_shutdown_listener(id_: UUID) -> bool:
    return _shutdown_listeners.pop(id_, None) is not None
