"""Centralized error handling for Slack integration.

This module provides:
- SlackErrorCode: Unique error codes for traceability
- SlackError: Exception class for user-facing errors
- get_user_message(): Function to get user-facing messages for error codes
"""

import logging
from enum import Enum
from typing import Any

from integrations.utils import HOST_URL

logger = logging.getLogger(__name__)


class SlackErrorCode(Enum):
    """Unique error codes for traceability in logs and user messages."""

    SESSION_EXPIRED = 'SLACK_ERR_001'
    REDIS_STORE_FAILED = 'SLACK_ERR_002'
    REDIS_RETRIEVE_FAILED = 'SLACK_ERR_003'
    USER_NOT_AUTHENTICATED = 'SLACK_ERR_004'
    PROVIDER_TIMEOUT = 'SLACK_ERR_005'
    PROVIDER_AUTH_FAILED = 'SLACK_ERR_006'
    LLM_AUTH_FAILED = 'SLACK_ERR_007'
    MISSING_SETTINGS = 'SLACK_ERR_008'
    UNEXPECTED_ERROR = 'SLACK_ERR_999'


class SlackError(Exception):
    """Exception for errors that should be communicated to the Slack user.

    This exception is caught by the centralized error handler in SlackManager,
    which logs the error and sends an appropriate message to the user.

    Usage:
        raise SlackError(SlackErrorCode.USER_NOT_AUTHENTICATED,
                        message_kwargs={'login_link': link})
    """

    def __init__(
        self,
        code: SlackErrorCode,
        message_kwargs: dict[str, Any] | None = None,
        log_context: dict[str, Any] | None = None,
    ):
        """Initialize a SlackError.

        Args:
            code: The error code identifying the type of error
            message_kwargs: Kwargs for formatting the user message
                           (e.g., {'login_link': '...'})
            log_context: Additional context for structured logging
        """
        self.code = code
        self.message_kwargs = message_kwargs or {}
        self.log_context = log_context or {}
        super().__init__(f'{code.value}: {code.name}')

    def get_user_message(self) -> str:
        """Get the user-facing message for this error."""
        return get_user_message(self.code, **self.message_kwargs)


# Centralized user-facing messages
_USER_MESSAGES: dict[SlackErrorCode, str] = {
    SlackErrorCode.SESSION_EXPIRED: (
        '⏰ Your session has expired. '
        'Please mention me again with your request to start a new conversation.'
    ),
    SlackErrorCode.REDIS_STORE_FAILED: (
        '⚠️ Something went wrong on our end (ref: {code}). '
        'Please try again in a few moments.'
    ),
    SlackErrorCode.REDIS_RETRIEVE_FAILED: (
        '⚠️ Something went wrong on our end (ref: {code}). '
        'Please try again in a few moments.'
    ),
    SlackErrorCode.USER_NOT_AUTHENTICATED: (
        '🔐 Please link your Slack account to OpenHands: '
        '[Click here to Login]({login_link})'
    ),
    SlackErrorCode.PROVIDER_TIMEOUT: (
        '⏱️ The request timed out while connecting to your git provider. '
        'Please try again.'
    ),
    SlackErrorCode.PROVIDER_AUTH_FAILED: (
        '🔐 Authentication with your git provider failed. '
        f'Please re-login at [OpenHands Cloud]({HOST_URL}) and try again.'
    ),
    SlackErrorCode.LLM_AUTH_FAILED: (
        '@{username} please set a valid LLM API key in '
        f'[OpenHands Cloud]({HOST_URL}) before starting a job.'
    ),
    SlackErrorCode.MISSING_SETTINGS: (
        '{username} please re-login into '
        f'[OpenHands Cloud]({HOST_URL}) before starting a job.'
    ),
    SlackErrorCode.UNEXPECTED_ERROR: (
        'Uh oh! There was an unexpected error (ref: {code}). Please try again later.'
    ),
}


def get_user_message(error_code: SlackErrorCode, **kwargs) -> str:
    """Get a user-facing message for a given error code.

    Args:
        error_code: The error code to get a message for
        **kwargs: Additional formatting arguments (e.g., username, login_link)

    Returns:
        Formatted user-facing message string
    """
    msg = _USER_MESSAGES.get(
        error_code, _USER_MESSAGES[SlackErrorCode.UNEXPECTED_ERROR]
    )
    try:
        return msg.format(code=error_code.value, **kwargs)
    except KeyError as e:
        logger.warning(
            f'Missing format key {e} in error message',
            extra={'error_code': error_code.value},
        )
        # Return a generic error message with the code for debugging
        return f'An error occurred (ref: {error_code.value}). Please try again later.'
