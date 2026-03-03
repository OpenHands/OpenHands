from server.auth.auth_error import AuthError, ExpiredError
from server.auth.saas_user_auth import saas_user_auth_from_signed_token
from server.auth.token_manager import TokenManager
from socketio.exceptions import ConnectionRefusedError
from storage.api_key_store import ApiKeyStore

from openhands.core.config import load_openhands_config
from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import ConversationStoreImpl
from openhands.storage.conversation.conversation_validator import ConversationValidator


class SaasConversationValidator(ConversationValidator):
    """Storage for conversation metadata. May or may not support multiple users depending on the environment."""

    async def _validate_api_key(self, api_key: str) -> str | None:
        """
        Validate an API key and return the user_id and github_user_id if valid.

        Args:
            api_key: The API key to validate

        Returns:
            A tuple of (user_id, github_user_id) if the API key is valid, None otherwise
        """
        try:
            token_manager = TokenManager()

            # Validate the API key and get the user_id
            api_key_store = ApiKeyStore.get_instance()
            user_id = await api_key_store.validate_api_key(api_key)

            if not user_id:
                logger.warning('Invalid API key')
                return None

            # Get the offline token for the user
            offline_token = await token_manager.load_offline_token(user_id)
            if not offline_token:
                logger.warning(f'No offline token found for user {user_id}')
                return None

            return user_id

        except Exception as e:
            logger.warning(f'Error validating API key: {str(e)}')
            return None

    async def _validate_conversation_access(
        self, conversation_id: str, user_id: str
    ) -> bool:
        """
        Validate that the user has access to the conversation.

        Args:
            conversation_id: The ID of the conversation
            user_id: The ID of the user
            github_user_id: The GitHub user ID, if available

        Returns:
            True if the user has access to the conversation, False otherwise

        Raises:
            ConnectionRefusedError: If the user does not have access to the conversation
        """
        config = load_openhands_config()
        conversation_store = await ConversationStoreImpl.get_instance(config, user_id)

        if not await conversation_store.validate_metadata(conversation_id, user_id):
            logger.error(
                f'User {user_id} is not allowed to join conversation {conversation_id}'
            )
            raise ConnectionRefusedError(
                f'User {user_id} is not allowed to join conversation {conversation_id}'
            )
        return True

    async def validate(
        self,
        conversation_id: str,
        cookies_str: str,
        authorization_header: str | None = None,
    ) -> str | None:
        """
        Validate the conversation access using either an API key from the Authorization header
        or a keycloak_auth cookie.

        Args:
            conversation_id: The ID of the conversation
            cookies_str: The cookies string from the request
            authorization_header: The Authorization header from the request, if available

        Returns:
            A tuple of (user_id, github_user_id)

        Raises:
            ConnectionRefusedError: If the user does not have access to the conversation
            AuthError: If the authentication fails
            RuntimeError: If there is an error with the configuration or user info
        """
        # Try to authenticate using Authorization header first
        if authorization_header and authorization_header.startswith('Bearer '):
            api_key = authorization_header.replace('Bearer ', '')
            user_id = await self._validate_api_key(api_key)

            if user_id:
                logger.info(
                    f'User {user_id} is connecting to conversation {conversation_id} via API key'
                )

                await self._validate_conversation_access(conversation_id, user_id)
                return user_id

        # Fall back to cookie authentication
        token_manager = TokenManager()
        config = load_openhands_config()
        cookies = (
            dict(cookie.split('=', 1) for cookie in cookies_str.split('; '))
            if cookies_str
            else {}
        )

        signed_token = cookies.get('keycloak_auth', '')
        if not signed_token:
            logger.warning('No keycloak_auth cookie or valid Authorization header')
            raise ConnectionRefusedError(
                'No keycloak_auth cookie or valid Authorization header'
            )
        if not config.jwt_secret:
            raise RuntimeError('JWT secret not found')

        try:
            user_auth = await saas_user_auth_from_signed_token(signed_token)
            access_token = await user_auth.get_access_token()
        except ExpiredError:
            raise ConnectionRefusedError('SESSION$TIMEOUT_MESSAGE')
        if access_token is None:
            raise AuthError('no_access_token')
        user_info_dict = await token_manager.get_user_info(
            access_token.get_secret_value()
        )
        if not user_info_dict or 'sub' not in user_info_dict:
            logger.info(
                f'Invalid user_info {user_info_dict} for access token {access_token}'
            )
            raise RuntimeError('Invalid user_info')
        user_id = user_info_dict['sub']

        logger.info(f'User {user_id} is connecting to conversation {conversation_id}')

        await self._validate_conversation_access(conversation_id, user_id)  # type: ignore
        return user_id
