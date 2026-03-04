import logging
from typing import Callable, Coroutine
from uuid import UUID

from integrations.utils import CONVERSATION_URL
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth.user_auth import UserAuth


def is_budget_exceeded_error(error_message: str) -> bool:
    """Check if an error message indicates a budget exceeded condition.

    This is used to downgrade error logs to info logs for budget exceeded errors
    since they are expected cost control behavior rather than unexpected errors.
    """
    lower_message = error_message.lower()
    return 'budget' in lower_message and 'exceeded' in lower_message


BUDGET_EXCEEDED_USER_MESSAGE = 'LLM budget has been exceeded, please re-fill.'


async def handle_callback_error(
    error: Exception,
    conversation_id: UUID,
    service_name: str,
    service_logger: logging.Logger,
    can_post_error: bool,
    post_error_func: Callable[[str], Coroutine],
) -> None:
    """Handle callback processing errors with appropriate logging and user messages.

    This centralizes the error handling logic for V1 callback processors to:
    - Log budget exceeded errors at INFO level (expected cost control behavior)
    - Log other errors at EXCEPTION level
    - Post user-friendly error messages to the integration platform

    Args:
        error: The exception that occurred
        conversation_id: The conversation ID for logging and linking
        service_name: The service name for log messages (e.g., "GitHub", "GitLab", "Slack")
        service_logger: The logger instance to use for logging
        can_post_error: Whether the prerequisites are met to post an error message
        post_error_func: Async function to post the error message to the platform
    """
    error_str = str(error)
    budget_exceeded = is_budget_exceeded_error(error_str)

    # Log appropriately based on error type
    if budget_exceeded:
        service_logger.info(
            '[%s V1] Budget exceeded for conversation %s: %s',
            service_name,
            conversation_id,
            error,
        )
    else:
        service_logger.exception(
            '[%s V1] Error processing callback: %s', service_name, error
        )

    # Try to post error message to the platform
    if can_post_error:
        try:
            error_detail = (
                BUDGET_EXCEEDED_USER_MESSAGE if budget_exceeded else error_str
            )
            await post_error_func(
                f'OpenHands encountered an error: **{error_detail}**\n\n'
                f'[See the conversation]({CONVERSATION_URL.format(conversation_id)}) '
                'for more information.'
            )
        except Exception as post_error:
            service_logger.warning(
                '[%s V1] Failed to post error message to %s: %s',
                service_name,
                service_name,
                post_error,
            )


async def get_saas_user_auth(
    keycloak_user_id: str, token_manager: TokenManager
) -> UserAuth:
    offline_token = await token_manager.load_offline_token(keycloak_user_id)
    if offline_token is None:
        logger.info('no_offline_token_found')

    user_auth = SaasUserAuth(
        user_id=keycloak_user_id,
        refresh_token=SecretStr(offline_token),
    )
    return user_auth
