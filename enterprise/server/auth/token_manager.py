import asyncio
import base64
import hashlib
import json
import time
from base64 import b64encode
from urllib.parse import parse_qs

import httpx
import jwt
from cryptography.fernet import Fernet
from jwt.exceptions import DecodeError
from keycloak.exceptions import (
    KeycloakAuthenticationError,
    KeycloakConnectionError,
    KeycloakError,
    KeycloakPostError,
)
from server.auth.auth_error import ExpiredError
from server.auth.constants import (
    BITBUCKET_APP_CLIENT_ID,
    BITBUCKET_APP_CLIENT_SECRET,
    DUPLICATE_EMAIL_CHECK,
    GITHUB_APP_CLIENT_ID,
    GITHUB_APP_CLIENT_SECRET,
    GITLAB_APP_CLIENT_ID,
    GITLAB_APP_CLIENT_SECRET,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL,
    KEYCLOAK_SERVER_URL_EXT,
)
from server.auth.email_validation import (
    extract_base_email,
    get_base_email_regex_pattern,
    matches_base_email,
)
from server.auth.keycloak_manager import get_keycloak_admin, get_keycloak_openid
from server.config import get_config
from server.logger import logger
from sqlalchemy import String as SQLString
from sqlalchemy import type_coerce
from storage.auth_token_store import AuthTokenStore
from storage.database import session_maker
from storage.github_app_installation import GithubAppInstallation
from storage.offline_token_store import OfflineTokenStore
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt

from openhands.integrations.service_types import ProviderType
from openhands.server.types import SessionExpiredError
from openhands.utils.http_session import httpx_verify_option


def _before_sleep_callback(retry_state: RetryCallState) -> None:
    logger.info(f'Retry attempt {retry_state.attempt_number} for Keycloak operation')


def create_encryption_utility(secret_key: bytes):
    """Creates an encryption utility using a 32-byte secret key.

    Args:
        secret_key (bytes): A 32-byte secret key
    Returns:
        tuple: (encrypt_string, decrypt_string) functions.
    """
    # Convert the 32-byte key into a Fernet key (32 bytes -> urlsafe base64)
    fernet_key = b64encode(hashlib.sha256(secret_key).digest())
    f = Fernet(fernet_key)

    def encrypt_text(text: str) -> str:
        return f.encrypt(text.encode()).decode()

    def encrypt_payload(payload: dict) -> str:
        """Encrypts a string and returns the result as a base64 string."""
        text = json.dumps(payload)
        return encrypt_text(text)

    def decrypt_text(encrypted_text: str) -> str:
        return f.decrypt(encrypted_text.encode()).decode()

    def decrypt_payload(encrypted_text: str) -> dict:
        """Decrypts a base64 encoded encrypted string."""
        text = decrypt_text(encrypted_text)
        return json.loads(text)

    return encrypt_payload, decrypt_payload, encrypt_text, decrypt_text


class TokenManager:
    def __init__(self, external: bool = False):
        self.external = external
        jwt_secret = get_config().jwt_secret.get_secret_value()
        (
            self.encrypt_payload,
            self.decrypt_payload,
            self.encrypt_text,
            self.decrypt_text,
        ) = create_encryption_utility(jwt_secret.encode())

    async def get_keycloak_tokens(
        self, code: str, redirect_uri: str
    ) -> tuple[str | None, str | None]:
        try:
            token_response = await get_keycloak_openid(self.external).a_token(
                grant_type='authorization_code',
                code=code,
                redirect_uri=redirect_uri,
            )

            logger.debug(f'token_response: {token_response}')

            if (
                'access_token' not in token_response
                or 'refresh_token' not in token_response
            ):
                logger.error('Missing either access or refresh token in response')
                return None, None

            return token_response['access_token'], token_response['refresh_token']
        except Exception:
            logger.exception('Exception when getting Keycloak tokens')
            return None, None

    async def verify_keycloak_token(
        self, keycloak_token: str, refresh_token: str
    ) -> tuple[str, str]:
        try:
            await get_keycloak_openid(self.external).a_userinfo(keycloak_token)
            return keycloak_token, refresh_token
        except KeycloakAuthenticationError:
            logger.debug('attempting to refresh keycloak access token')
            new_keycloak_tokens = await get_keycloak_openid(
                self.external
            ).a_refresh_token(refresh_token)
            logger.info('Refreshed keycloak access token')
            return (
                new_keycloak_tokens['access_token'],
                new_keycloak_tokens['refresh_token'],
            )

    # UserInfo from Keycloak return a dictionary with the following format:
    # {
    # 'sub': '248289761001',
    # 'name': 'Jane Doe',
    # 'given_name': 'Jane',
    # 'family_name': 'Doe',
    # 'preferred_username': 'j.doe',
    # 'email': 'janedoe@example.com',
    # 'picture': 'http://example.com/janedoe/me.jpg'
    # 'github_id': '354322532'
    # }
    async def get_user_info(self, access_token: str) -> dict:
        if not access_token:
            return {}
        user_info = await get_keycloak_openid(self.external).a_userinfo(access_token)
        return user_info

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(KeycloakConnectionError),
        before_sleep=_before_sleep_callback,
    )
    async def store_idp_tokens(
        self,
        idp: ProviderType,
        user_id: str,
        keycloak_access_token: str,
    ):
        data = await self.get_idp_tokens_from_keycloak(keycloak_access_token, idp)
        if data:
            await self._store_idp_tokens(
                user_id,
                idp,
                str(data['access_token']),
                str(data['refresh_token']),
                int(data['access_token_expires_at']),
                int(data['refresh_token_expires_at']),
            )

    async def _store_idp_tokens(
        self,
        user_id: str,
        identity_provider: ProviderType,
        access_token: str,
        refresh_token: str,
        access_token_expires_at: int,
        refresh_token_expires_at: int,
    ):
        token_store = await AuthTokenStore.get_instance(
            keycloak_user_id=user_id, idp=identity_provider
        )
        encrypted_access_token = self.encrypt_text(access_token)
        encrypted_refresh_token = self.encrypt_text(refresh_token)
        await token_store.store_tokens(
            encrypted_access_token,
            encrypted_refresh_token,
            access_token_expires_at,
            refresh_token_expires_at,
        )

    async def get_idp_tokens_from_keycloak(
        self,
        access_token: str,
        idp: ProviderType,
    ) -> dict[str, str | int]:
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            base_url = KEYCLOAK_SERVER_URL_EXT if self.external else KEYCLOAK_SERVER_URL
            url = f'{base_url}/realms/{KEYCLOAK_REALM_NAME}/broker/{idp.value}/token'
            headers = {
                'Authorization': f'Bearer {access_token}',
            }

            data: dict[str, str | int] = {}
            response = await client.get(url, headers=headers)
            content_str = response.content.decode('utf-8')
            if (
                f'Identity Provider [{idp.value}] does not support this operation.'
                in content_str
            ):
                return data
            response.raise_for_status()
            try:
                # Try parsing as JSON
                data = json.loads(response.text)
            except json.JSONDecodeError:
                # If it's not JSON, try parsing as a URL-encoded string
                parsed = parse_qs(response.text)
                # Convert lists to strings and specific keys to integers
                data = {
                    key: int(value[0])
                    if key
                    in {'expires_in', 'refresh_token_expires_in', 'refresh_expires_in'}
                    else value[0]
                    for key, value in parsed.items()
                }

            current_time = int(time.time())
            expires_in = int(data.get('expires_in', 0))
            refresh_expires_in = int(
                data.get('refresh_token_expires_in', data.get('refresh_expires_in', 0))
            )
            access_token_expires_at = (
                0 if expires_in == 0 else current_time + expires_in
            )
            refresh_token_expires_at = (
                0 if refresh_expires_in == 0 else current_time + refresh_expires_in
            )

            return {
                'access_token': data['access_token'],
                'refresh_token': data['refresh_token'],
                'access_token_expires_at': access_token_expires_at,
                'refresh_token_expires_at': refresh_token_expires_at,
            }

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(KeycloakConnectionError),
        before_sleep=_before_sleep_callback,
    )
    async def get_idp_token(
        self,
        access_token: str,
        idp: ProviderType,
    ) -> str:
        # Get user info to determine user_id and idp
        user_info = await self.get_user_info(access_token=access_token)
        user_id = user_info.get('sub')
        username = user_info.get('preferred_username')
        logger.info(f'Getting token for user {username} and IDP {idp}')
        token_store = await AuthTokenStore.get_instance(
            keycloak_user_id=user_id, idp=idp
        )

        try:
            token_info = await token_store.load_tokens(
                self._check_expiration_and_refresh
            )
            if not token_info:
                logger.info(f'No tokens for user: {username}, identity provider: {idp}')
                raise ValueError(
                    f'No tokens for user: {username}, identity provider: {idp}'
                )
            access_token = self.decrypt_text(token_info['access_token'])
            logger.info(f'Got {idp} token: {access_token[0:5]}')
            return access_token
        except httpx.HTTPStatusError as e:
            # Log the full response details including the body
            logger.error(
                f'Failed to get tokens for user {username}, identity provider {idp} from URL {e.response.url}. '
                f'Status code: {e.response.status_code}, '
                f'Response body: {e.response.text}'
            )
            raise ValueError(
                f'Failed to get token for user: {username}, identity provider: {idp}. '
                f'Status code: {e.response.status_code}, '
                f'Response body: {e.response.text}'
            ) from e

    async def _check_expiration_and_refresh(
        self,
        identity_provider: ProviderType,
        encrypted_refresh_token: str,
        access_token_expires_at: int,
        refresh_token_expires_at: int,
    ) -> dict[str, str | int] | None:
        current_time = int(time.time())
        # expire access_token four hours before actual expiration
        # This ensures tokens are refreshed on resume to have at least 4 hours validity
        access_expired = (
            False
            if access_token_expires_at == 0
            else access_token_expires_at < current_time + 14400
        )
        refresh_expired = (
            False
            if refresh_token_expires_at == 0
            else refresh_token_expires_at < current_time
        )

        if not access_expired:
            return None
        if access_expired and refresh_expired:
            logger.error('Both Access and Refresh Tokens expired.')
            raise ValueError('Both Access and Refresh Tokens expired.')

        logger.info(f'Access token expired for {identity_provider}. Refreshing token.')
        refresh_token = self.decrypt_text(encrypted_refresh_token)
        token_data = await self._refresh_token(identity_provider, refresh_token)
        access_token = token_data['access_token']
        refresh_token = token_data['refresh_token']
        access_expiration = token_data['access_token_expires_at']
        refresh_expiration = token_data['refresh_token_expires_at']

        return {
            'access_token': self.encrypt_text(access_token),
            'refresh_token': self.encrypt_text(refresh_token),
            'access_token_expires_at': access_expiration,
            'refresh_token_expires_at': refresh_expiration,
        }

    async def _refresh_token(
        self, idp: ProviderType, refresh_token: str
    ) -> dict[str, str | int]:
        logger.info(f'Refreshing {idp} token')
        if idp == ProviderType.GITHUB:
            return await self._refresh_github_token(refresh_token)
        elif idp == ProviderType.GITLAB:
            return await self._refresh_gitlab_token(refresh_token)
        elif idp == ProviderType.BITBUCKET:
            return await self._refresh_bitbucket_token(refresh_token)
        else:
            raise ValueError(f'Unsupported IDP: {idp}')

    async def _refresh_github_token(self, refresh_token: str) -> dict[str, str | int]:
        url = 'https://github.com/login/oauth/access_token'
        logger.info(f'Refreshing GitHub token with URL: {url}')

        payload = {
            'client_id': GITHUB_APP_CLIENT_ID,
            'client_secret': GITHUB_APP_CLIENT_SECRET,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token',
        }
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()
            logger.info('Successfully refreshed GitHub token')
            parsed = parse_qs(response.text)

            # Convert lists to strings and specific keys to integers
            data = {
                key: int(value[0])
                if key
                in {'expires_in', 'refresh_token_expires_in', 'refresh_expires_in'}
                else value[0]
                for key, value in parsed.items()
            }
            return await self._parse_refresh_response(data)

    async def _refresh_gitlab_token(self, refresh_token: str) -> dict[str, str | int]:
        url = 'https://gitlab.com/oauth/token'
        logger.info(f'Refreshing GitLab token with URL: {url}')

        payload = {
            'client_id': GITLAB_APP_CLIENT_ID,
            'client_secret': GITLAB_APP_CLIENT_SECRET,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token',
        }
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()
            logger.info('Successfully refreshed GitLab token')

            data = response.json()
            return await self._parse_refresh_response(data)

    async def _refresh_bitbucket_token(
        self, refresh_token: str
    ) -> dict[str, str | int]:
        url = 'https://bitbucket.org/site/oauth2/access_token'
        logger.info(f'Refreshing Bitbucket token with URL: {url}')

        auth = base64.b64encode(
            f'{BITBUCKET_APP_CLIENT_ID}:{BITBUCKET_APP_CLIENT_SECRET}'.encode()
        ).decode()

        headers = {
            'Authorization': f'Basic {auth}',
            'Content-Type': 'application/x-www-form-urlencoded',
        }

        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
        }

        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, data=data, headers=headers)
            response.raise_for_status()
            logger.info('Successfully refreshed Bitbucket token')

            data = response.json()
            return await self._parse_refresh_response(data)

    async def _parse_refresh_response(self, data: dict) -> dict[str, str | int]:
        access_token = data.get('access_token')
        refresh_token = data.get('refresh_token')
        if not access_token or not refresh_token:
            if data.get('error') == 'bad_refresh_token':
                raise ExpiredError()
            raise ValueError(
                'Failed to refresh token: missing access_token or refresh_token in response.'
            )

        expires_in = int(data.get('expires_in', 0))
        refresh_expires_in = int(
            data.get('refresh_token_expires_in', data.get('refresh_expires_in', 0))
        )
        current_time = int(time.time())
        access_token_expires_at = 0 if expires_in == 0 else current_time + expires_in
        refresh_token_expires_at = (
            0 if refresh_expires_in == 0 else current_time + refresh_expires_in
        )

        logger.info(
            f'Token refresh successful. New access token expires at: {access_token_expires_at}, refresh token expires at: {refresh_token_expires_at}'
        )
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'access_token_expires_at': access_token_expires_at,
            'refresh_token_expires_at': refresh_token_expires_at,
        }

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(KeycloakConnectionError),
        before_sleep=_before_sleep_callback,
    )
    async def get_idp_token_from_offline_token(
        self, offline_token: str, idp: ProviderType
    ) -> str:
        logger.info('Getting IDP token from offline token')

        try:
            tokens = await get_keycloak_openid(self.external).a_refresh_token(
                offline_token
            )
            return await self.get_idp_token(tokens['access_token'], idp)
        except KeycloakConnectionError:
            logger.exception('KeycloakConnectionError when refreshing token')
            raise
        except KeycloakPostError as e:
            error_message = str(e)
            if 'invalid_grant' in error_message or 'session not found' in error_message:
                logger.warning(f'User session expired or invalid: {error_message}')
                raise SessionExpiredError(
                    'Your session has expired. Please login again.'
                ) from e
            raise

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(KeycloakConnectionError),
        before_sleep=_before_sleep_callback,
    )
    async def get_idp_token_from_idp_user_id(
        self, idp_user_id: str, idp: ProviderType
    ) -> str | None:
        logger.info(f'Getting IDP token from IDP user_id: {idp_user_id}')
        user_id = await self.get_user_id_from_idp_user_id(idp_user_id, idp)
        if not user_id:
            return None

        try:
            offline_token = await self.load_offline_token(user_id=user_id)
            if not offline_token:
                logger.warning(f'No offline token found for user_id: {user_id}')
                return None
            return await self.get_idp_token_from_offline_token(
                offline_token=offline_token, idp=idp
            )
        except KeycloakConnectionError as e:
            logger.exception(
                f'KeycloakConnectionError when getting IDP token for IDP user_id {idp_user_id}: {str(e)}'
            )
            raise

    async def get_user_id_from_idp_user_id(
        self, idp_user_id: str, idp: ProviderType
    ) -> str | None:
        keycloak_admin = get_keycloak_admin(self.external)
        users = await keycloak_admin.a_get_users({'q': f'{idp.value}_id:{idp_user_id}'})
        if not users:
            logger.info(f'{idp.value} user with IDP ID {idp_user_id} not found.')
            return None
        keycloak_user_id = users[0]['id']
        logger.info(f'Got user ID {keycloak_user_id} from IDP user ID: {idp_user_id}')
        return keycloak_user_id

    async def get_user_id_from_user_email(self, email: str) -> str | None:
        keycloak_admin = get_keycloak_admin(self.external)
        users = await keycloak_admin.a_get_users({'q': f'email:{email}'})
        if not users:
            logger.error(f'User with email {email} not found.')
            return None
        keycloak_user_id = users[0]['id']
        logger.info(f'Got user ID {keycloak_user_id} from email: {email}')
        return keycloak_user_id

    async def _query_users_by_wildcard_pattern(
        self, local_part: str, domain: str
    ) -> dict[str, dict]:
        """Query Keycloak for users matching a wildcard email pattern.

        Tries multiple query methods to find users with emails matching
        the pattern {local_part}*@{domain}. This catches the base email
        and all + modifier variants.

        Args:
            local_part: The local part of the email (before @)
            domain: The domain part of the email (after @)

        Returns:
            Dictionary mapping user IDs to user objects
        """
        keycloak_admin = get_keycloak_admin(self.external)
        all_users = {}

        # Query for users with emails matching the base pattern using wildcard
        # Pattern: {local_part}*@{domain} - catches base email and all + variants
        # This may also catch unintended matches (e.g., joesmith@example.com), but
        # they will be filtered out by the regex pattern check later
        # Use 'search' parameter for Keycloak 26+ (better wildcard support)
        wildcard_queries = [
            {'search': f'{local_part}*@{domain}'},  # Try 'search' parameter first
            {'q': f'email:{local_part}*@{domain}'},  # Fallback to 'q' parameter
        ]

        for query_params in wildcard_queries:
            try:
                users = await keycloak_admin.a_get_users(query_params)
                for user in users:
                    all_users[user.get('id')] = user
                break  # Success, no need to try fallback
            except Exception as e:
                logger.debug(
                    f'Wildcard query failed with {list(query_params.keys())[0]}: {e}'
                )
                continue  # Try next query method

        return all_users

    def _find_duplicate_in_users(
        self, users: dict[str, dict], base_email: str, current_user_id: str
    ) -> bool:
        """Check if any user in the provided list matches the base email pattern.

        Filters users to find duplicates that match the base email pattern,
        excluding the current user.

        Args:
            users: Dictionary mapping user IDs to user objects
            base_email: The base email to match against
            current_user_id: The user ID to exclude from the check

        Returns:
            True if a duplicate is found, False otherwise
        """
        regex_pattern = get_base_email_regex_pattern(base_email)
        if not regex_pattern:
            logger.warning(
                f'Could not generate regex pattern for base email: {base_email}'
            )
            # Fallback to simple matching
            for user in users.values():
                user_email = user.get('email', '').lower()
                if (
                    user_email
                    and user.get('id') != current_user_id
                    and matches_base_email(user_email, base_email)
                ):
                    logger.info(
                        f'Found duplicate email: {user_email} matches base {base_email}'
                    )
                    return True
        else:
            for user in users.values():
                user_email = user.get('email', '')
                if (
                    user_email
                    and user.get('id') != current_user_id
                    and regex_pattern.match(user_email)
                ):
                    logger.info(
                        f'Found duplicate email: {user_email} matches base {base_email}'
                    )
                    return True

        return False

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(KeycloakConnectionError),
        before_sleep=_before_sleep_callback,
    )
    async def check_duplicate_base_email(
        self, email: str, current_user_id: str
    ) -> bool:
        """Check if a user with the same base email already exists.

        This method checks for duplicate signups using email + modifier.
        It checks if any user exists with the same base email, regardless of whether
        the provided email has a + modifier or not.

        Examples:
            - If email is "joe+test@example.com", it checks for existing users with
              base email "joe@example.com" (e.g., "joe@example.com", "joe+1@example.com")
            - If email is "joe@example.com", it checks for existing users with
              base email "joe@example.com" (e.g., "joe+1@example.com", "joe+test@example.com")

        Args:
            email: The email address to check (may or may not contain + modifier)
            current_user_id: The user ID of the current user (to exclude from check)

        Returns:
            True if a duplicate is found (excluding current user), False otherwise
        """
        if not email:
            return False

        # We have the option to skip the duplicate email check in test environments
        if not DUPLICATE_EMAIL_CHECK:
            return False

        base_email = extract_base_email(email)
        if not base_email:
            logger.warning(f'Could not extract base email from: {email}')
            return False

        try:
            local_part, domain = base_email.rsplit('@', 1)
            users = await self._query_users_by_wildcard_pattern(local_part, domain)
            return self._find_duplicate_in_users(users, base_email, current_user_id)

        except KeycloakConnectionError:
            logger.exception('KeycloakConnectionError when checking duplicate email')
            raise
        except Exception as e:
            logger.exception(f'Unexpected error checking duplicate email: {e}')
            # On any error, allow signup to proceed (fail open)
            return False

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(KeycloakConnectionError),
        before_sleep=_before_sleep_callback,
    )
    async def delete_keycloak_user(self, user_id: str) -> bool:
        """Delete a user from Keycloak.

        This method is used to clean up user accounts that were created
        but should not exist (e.g., duplicate email signups).

        Args:
            user_id: The Keycloak user ID to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            keycloak_admin = get_keycloak_admin(self.external)
            # Use the sync method (python-keycloak doesn't have async delete_user)
            # Run it in a thread executor to avoid blocking the event loop
            await asyncio.to_thread(keycloak_admin.delete_user, user_id)
            logger.info(f'Successfully deleted Keycloak user {user_id}')
            return True
        except KeycloakConnectionError:
            logger.exception(f'KeycloakConnectionError when deleting user {user_id}')
            raise
        except KeycloakError as e:
            # User might not exist or already deleted
            logger.warning(
                f'KeycloakError when deleting user {user_id}: {e}',
                extra={'user_id': user_id, 'error': str(e)},
            )
            return False
        except Exception as e:
            logger.exception(f'Unexpected error deleting Keycloak user {user_id}: {e}')
            return False

    async def get_user_info_from_user_id(self, user_id: str) -> dict | None:
        keycloak_admin = get_keycloak_admin(self.external)
        user = await keycloak_admin.a_get_user(user_id)
        if not user:
            logger.error(f'User with ID {user_id} not found.')
            return None
        return user

    async def get_github_id_from_user_id(self, user_id: str) -> str | None:
        user_info = await self.get_user_info_from_user_id(user_id)
        if user_info is None:
            return None
        github_ids = (user_info.get('attributes') or {}).get('github_id')
        if not github_ids:
            return None
        github_id = github_ids[0]
        return github_id

    async def disable_keycloak_user(
        self, user_id: str, email: str | None = None
    ) -> None:
        """Disable a Keycloak user account.

        Args:
            user_id: The Keycloak user ID to disable
            email: Optional email address for logging purposes

        This method attempts to disable the user account but will not raise exceptions.
        Errors are logged but do not prevent the operation from completing.
        """
        try:
            keycloak_admin = get_keycloak_admin(self.external)
            # Get current user to preserve other fields
            user = await keycloak_admin.a_get_user(user_id)
            if user:
                # Update user with enabled=False to disable the account
                await keycloak_admin.a_update_user(
                    user_id=user_id,
                    payload={
                        'enabled': False,
                        'username': user.get('username', ''),
                        'email': user.get('email', ''),
                        'emailVerified': user.get('emailVerified', False),
                    },
                )
                email_str = f', email: {email}' if email else ''
                logger.info(
                    f'Disabled Keycloak account for user_id: {user_id}{email_str}'
                )
            else:
                logger.warning(
                    f'User not found in Keycloak when attempting to disable: {user_id}'
                )
        except Exception as e:
            # Log error but don't raise - the caller should handle the blocking regardless
            email_str = f', email: {email}' if email else ''
            logger.error(
                f'Failed to disable Keycloak account for user_id: {user_id}{email_str}: {str(e)}',
                exc_info=True,
            )

    def store_org_token(self, installation_id: int, installation_token: str):
        """Store a GitHub App installation token.

        Args:
            installation_id: GitHub installation ID (integer or string)
            installation_token: The token to store
        """
        with session_maker() as session:
            # Ensure installation_id is a string
            str_installation_id = str(installation_id)
            # Use type_coerce to ensure SQLAlchemy treats the parameter as a string
            installation = (
                session.query(GithubAppInstallation)
                .filter(
                    GithubAppInstallation.installation_id
                    == type_coerce(str_installation_id, SQLString)
                )
                .first()
            )
            if installation:
                installation.encrypted_token = self.encrypt_text(installation_token)
            else:
                session.add(
                    GithubAppInstallation(
                        installation_id=str_installation_id,  # Use the string version
                        encrypted_token=self.encrypt_text(installation_token),
                    )
                )
            session.commit()

    def load_org_token(self, installation_id: int) -> str | None:
        """Load a GitHub App installation token.

        Args:
            installation_id: GitHub installation ID (integer or string)

        Returns:
            The decrypted token if found, None otherwise
        """
        with session_maker() as session:
            # Ensure installation_id is a string and use type_coerce
            str_installation_id = str(installation_id)
            installation = (
                session.query(GithubAppInstallation)
                .filter(
                    GithubAppInstallation.installation_id
                    == type_coerce(str_installation_id, SQLString)
                )
                .first()
            )
            if not installation:
                return None
            token = self.decrypt_text(installation.encrypted_token)
            return token

    async def store_offline_token(self, user_id: str, offline_token: str):
        token_store = await OfflineTokenStore.get_instance(get_config(), user_id)
        encrypted_tokens = self.encrypt_payload({'refresh_token': offline_token})
        payload = {'tokens': encrypted_tokens}
        await token_store.store_token(json.dumps(payload))

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(KeycloakConnectionError),
        before_sleep=_before_sleep_callback,
    )
    async def refresh(self, refresh_token: str) -> dict:
        try:
            return await get_keycloak_openid(self.external).a_refresh_token(
                refresh_token
            )
        except KeycloakError as e:
            try:
                # We can log the token payload without the signature
                refresh_token_payload = jwt.decode(
                    refresh_token, options={'verify_signature': False}
                )
                logger.info(
                    'error_with_refresh_token',
                    extra={
                        'refresh_token': refresh_token_payload,
                        'error': str(e),
                    },
                )
            except DecodeError:
                # Whatever was passed in as a refresh token was completely wrong.
                # We can log this on the basis of it not being a real secret.
                logger.info(
                    'refresh_token_was_not_a_jwt',
                    extra={'refresh_token': refresh_token},
                )
            raise

    async def validate_offline_token(self, user_id: str) -> bool:
        offline_token = await self.load_offline_token(user_id=user_id)
        if not offline_token:
            return False

        validated = False
        try:
            await get_keycloak_openid(self.external).a_refresh_token(offline_token)
            validated = True
        except KeycloakError:
            pass

        return validated

    async def check_offline_token_is_active(self, user_id: str) -> bool:
        offline_token = await self.load_offline_token(user_id=user_id)
        if not offline_token:
            return False

        active = False
        try:
            token_info = await get_keycloak_openid(self.external).a_introspect(
                offline_token
            )
            if token_info.get('active'):
                active = True
        except KeycloakError:
            pass

        return active

    async def load_offline_token(self, user_id: str) -> str | None:
        token_store = await OfflineTokenStore.get_instance(get_config(), user_id)
        payload = await token_store.load_token()
        if not payload:
            return None
        cred = json.loads(payload)
        encrypted_tokens = cred['tokens']
        tokens = self.decrypt_payload(encrypted_tokens)
        return tokens['refresh_token']

    async def logout(self, refresh_token: str):
        try:
            await get_keycloak_openid(self.external).a_logout(
                refresh_token=refresh_token
            )
        except Exception:
            logger.exception('Exception when logging out of keycloak')
            raise
