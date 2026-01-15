# DEPRECATED: This module is part of the deprecated 'openhands.runtime' package.
# It will be removed on April 1, 2025. Please migrate to the OpenHands SDK:
# https://github.com/All-Hands-AI/openhands-sdk
from tenacity import RetryCallState
from tenacity.stop import stop_base

from openhands.utils.shutdown_listener import should_exit


class stop_if_should_exit(stop_base):
    """Stop if the should_exit flag is set."""

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        return should_exit()
