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
