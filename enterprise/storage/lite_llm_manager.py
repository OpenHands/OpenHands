"""
Store class for managing organizational settings.
"""

import functools
import os
from typing import Any, Awaitable, Callable

import httpx
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.constants import (
    DEFAULT_INITIAL_BUDGET,
    LITE_LLM_API_KEY,
    LITE_LLM_API_URL,
    LITE_LLM_TEAM_ID,
    ORG_SETTINGS_VERSION,
    get_default_litellm_model,
)
from server.logger import logger
from storage.user_settings import UserSettings

from openhands.server.settings import Settings
from openhands.utils.http_session import httpx_verify_option

# Timeout in seconds for BYOR key verification requests to LiteLLM
BYOR_KEY_VERIFICATION_TIMEOUT = 5.0

# A very large number to represent "unlimited" until LiteLLM fixes their unlimited update bug.
UNLIMITED_BUDGET_SETTING = 1000000000.0


class LiteLlmManager:
    """Manage LiteLLM interactions."""

    @staticmethod
    async def create_entries(
        org_id: str,
        keycloak_user_id: str,
        oss_settings: Settings,
        create_user: bool,
    ) -> Settings | None:
        logger.info(
            'SettingsStore:update_settings_with_litellm_default:start',
            extra={'org_id': org_id, 'user_id': keycloak_user_id},
        )
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        local_deploy = os.environ.get('LOCAL_DEPLOYMENT', None)
        key = LITE_LLM_API_KEY
        if not local_deploy:
            # Get user info to add to litellm
            token_manager = TokenManager()
            keycloak_user_info = (
                await token_manager.get_user_info_from_user_id(keycloak_user_id) or {}
            )

            async with httpx.AsyncClient(
                headers={
                    'x-goog-api-key': LITE_LLM_API_KEY,
                }
            ) as client:
                await LiteLlmManager._create_team(
                    client, keycloak_user_id, org_id, DEFAULT_INITIAL_BUDGET
                )

                if create_user:
                    await LiteLlmManager._create_user(
                        client, keycloak_user_info.get('email'), keycloak_user_id
                    )

                await LiteLlmManager._add_user_to_team(
                    client, keycloak_user_id, org_id, DEFAULT_INITIAL_BUDGET
                )

                key = await LiteLlmManager._generate_key(
                    client,
                    keycloak_user_id,
                    org_id,
                    f'OpenHands Cloud - user {keycloak_user_id} - org {org_id}',
                    None,
                )

        oss_settings.agent = 'CodeActAgent'
        # Use the model corresponding to the current user settings version
        oss_settings.llm_model = get_default_litellm_model()
        oss_settings.llm_api_key = SecretStr(key)
        oss_settings.llm_base_url = LITE_LLM_API_URL
        return oss_settings

    @staticmethod
    async def migrate_entries(
        org_id: str,
        keycloak_user_id: str,
        user_settings: UserSettings,
    ) -> UserSettings | None:
        logger.info(
            'LiteLlmManager:migrate_lite_llm_entries:start',
            extra={'org_id': org_id, 'user_id': keycloak_user_id},
        )
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        local_deploy = os.environ.get('LOCAL_DEPLOYMENT', None)
        if not local_deploy:
            # Get user info to add to litellm
            async with httpx.AsyncClient(
                headers={
                    'x-goog-api-key': LITE_LLM_API_KEY,
                }
            ) as client:
                user_json = await LiteLlmManager._get_user(client, keycloak_user_id)
                if not user_json:
                    return None
                user_info = user_json['user_info']
                max_budget = user_info.get('max_budget', 0.0)
                spend = user_info.get('spend', 0.0)
                # In upgrade to V4, we no longer use billing margin, but instead apply this directly
                # in litellm. The default billing marign was 2 before this (hence the magic numbers below)
                if (
                    user_settings
                    and user_settings.user_version < 4
                    and user_settings.billing_margin
                    and user_settings.billing_margin != 1.0
                ):
                    billing_margin = user_settings.billing_margin
                    logger.info(
                        'user_settings_v4_budget_upgrade',
                        extra={
                            'max_budget': max_budget,
                            'billing_margin': billing_margin,
                            'spend': spend,
                        },
                    )
                    max_budget *= billing_margin
                    spend *= billing_margin

                if not max_budget:
                    # if max_budget is None, then we've already migrated the User
                    return None
                credits = max(max_budget - spend, 0.0)

                logger.debug(
                    'LiteLlmManager:migrate_lite_llm_entries:create_team',
                    extra={'org_id': org_id, 'user_id': keycloak_user_id},
                )
                await LiteLlmManager._create_team(
                    client, keycloak_user_id, org_id, credits
                )

                logger.debug(
                    'LiteLlmManager:migrate_lite_llm_entries:update_user',
                    extra={'org_id': org_id, 'user_id': keycloak_user_id},
                )
                await LiteLlmManager._update_user(
                    client, keycloak_user_id, max_budget=UNLIMITED_BUDGET_SETTING
                )

                logger.debug(
                    'LiteLlmManager:migrate_lite_llm_entries:add_user_to_team',
                    extra={'org_id': org_id, 'user_id': keycloak_user_id},
                )
                await LiteLlmManager._add_user_to_team(
                    client, keycloak_user_id, org_id, credits
                )

                if user_settings.llm_api_key:
                    logger.debug(
                        'LiteLlmManager:migrate_lite_llm_entries:update_key',
                        extra={'org_id': org_id, 'user_id': keycloak_user_id},
                    )
                    await LiteLlmManager._update_key(
                        client,
                        keycloak_user_id,
                        user_settings.llm_api_key,
                        team_id=org_id,
                    )

                if user_settings.llm_api_key_for_byor:
                    logger.debug(
                        'LiteLlmManager:migrate_lite_llm_entries:update_byor_key',
                        extra={'org_id': org_id, 'user_id': keycloak_user_id},
                    )
                    await LiteLlmManager._update_key(
                        client,
                        keycloak_user_id,
                        user_settings.llm_api_key_for_byor,
                        team_id=org_id,
                    )

        logger.info(
            'LiteLlmManager:migrate_lite_llm_entries:complete',
            extra={'org_id': org_id, 'user_id': keycloak_user_id},
        )
        return user_settings

    @staticmethod
    async def update_team_and_users_budget(
        team_id: str,
        max_budget: float,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        async with httpx.AsyncClient(
            headers={
                'x-goog-api-key': LITE_LLM_API_KEY,
            }
        ) as client:
            await LiteLlmManager._update_team(client, team_id, None, max_budget)
            team_info = await LiteLlmManager._get_team(client, team_id)
            if not team_info:
                return None
            # TODO: change to use bulk update endpoint
            for membership in team_info.get('team_memberships', []):
                user_id = membership.get('user_id')
                if not user_id:
                    continue
                await LiteLlmManager._update_user_in_team(
                    client, user_id, team_id, max_budget
                )

    @staticmethod
    async def _create_team(
        client: httpx.AsyncClient,
        team_alias: str,
        team_id: str,
        max_budget: float,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/team/new',
            json={
                'team_id': team_id,
                'team_alias': team_alias,
                'models': [],
                'max_budget': max_budget,
                'spend': 0,
                'metadata': {
                    'version': ORG_SETTINGS_VERSION,
                    'model': get_default_litellm_model(),
                },
            },
        )
        # Team failed to create in litellm - this is an unforseen error state...
        if not response.is_success:
            if (
                response.status_code == 400
                and 'already exists. Please use a different team id' in response.text
            ):
                # team already exists, so update, then return
                await LiteLlmManager._update_team(
                    client, team_id, team_alias, max_budget
                )
                return
            logger.error(
                'error_creating_litellm_team',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'team_id': team_id,
                    'max_budget': max_budget,
                },
            )
        response.raise_for_status()

    @staticmethod
    async def _get_team(client: httpx.AsyncClient, team_id: str) -> dict | None:
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        """Get a team from litellm with the id matching that given."""
        response = await client.get(
            f'{LITE_LLM_API_URL}/team/info?team_id={team_id}',
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    async def _update_team(
        client: httpx.AsyncClient,
        team_id: str,
        team_alias: str | None,
        max_budget: float | None,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        json_data: dict[str, Any] = {
            'team_id': team_id,
            'metadata': {
                'version': ORG_SETTINGS_VERSION,
                'model': get_default_litellm_model(),
            },
        }

        if max_budget is not None:
            json_data['max_budget'] = max_budget

        if team_alias is not None:
            json_data['team_alias'] = team_alias

        response = await client.post(
            f'{LITE_LLM_API_URL}/team/update',
            json=json_data,
        )

        # Team failed to update in litellm - this is an unforseen error state...
        if not response.is_success:
            logger.error(
                'error_updating_litellm_team',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'team_id': [team_id],
                    'max_budget': max_budget,
                },
            )
        response.raise_for_status()

    @staticmethod
    async def _create_user(
        client: httpx.AsyncClient,
        email: str | None,
        keycloak_user_id: str,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/user/new',
            json={
                'user_email': email,
                'models': [],
                'user_id': keycloak_user_id,
                'teams': [LITE_LLM_TEAM_ID],
                'auto_create_key': False,
                'send_invite_email': False,
                'metadata': {
                    'version': ORG_SETTINGS_VERSION,
                    'model': get_default_litellm_model(),
                },
            },
        )
        if not response.is_success:
            logger.warning(
                'duplicate_user_email',
                extra={
                    'user_id': keycloak_user_id,
                    'email': email,
                },
            )
            # Litellm insists on unique email addresses - it is possible the email address was registered with a different user.
            response = await client.post(
                f'{LITE_LLM_API_URL}/user/new',
                json={
                    'user_email': None,
                    'models': [],
                    'user_id': keycloak_user_id,
                    'teams': [LITE_LLM_TEAM_ID],
                    'auto_create_key': False,
                    'send_invite_email': False,
                    'metadata': {
                        'version': ORG_SETTINGS_VERSION,
                        'model': get_default_litellm_model(),
                    },
                },
            )

            # User failed to create in litellm - this is an unforseen error state...
            if not response.is_success:
                if (
                    response.status_code in (400, 409)
                    and 'already exists' in response.text
                ):
                    logger.warning(
                        'litellm_user_already_exists',
                        extra={
                            'user_id': keycloak_user_id,
                        },
                    )
                    return
                logger.error(
                    'error_creating_litellm_user',
                    extra={
                        'status_code': response.status_code,
                        'text': response.text,
                        'user_id': [keycloak_user_id],
                        'email': None,
                    },
                )
            response.raise_for_status()

    @staticmethod
    async def _get_user(client: httpx.AsyncClient, user_id: str) -> dict | None:
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        """Get a user from litellm with the id matching that given."""
        response = await client.get(
            f'{LITE_LLM_API_URL}/user/info?user_id={user_id}',
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    async def _update_user(
        client: httpx.AsyncClient,
        keycloak_user_id: str,
        **kwargs,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return

        payload = {
            'user_id': keycloak_user_id,
        }
        payload.update(kwargs)

        response = await client.post(
            f'{LITE_LLM_API_URL}/user/update',
            json=payload,
        )

        if not response.is_success:
            logger.error(
                'error_updating_litellm_user',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'user_id': keycloak_user_id,
                },
            )
        response.raise_for_status()

    @staticmethod
    async def _update_key(
        client: httpx.AsyncClient,
        keycloak_user_id: str,
        key: str,
        **kwargs,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return

        payload = {
            'key': key,
        }
        payload.update(kwargs)

        response = await client.post(
            f'{LITE_LLM_API_URL}/key/update',
            json=payload,
        )

        if not response.is_success:
            if response.status_code == 401:
                logger.warning(
                    'invalid_litellm_key_during_update',
                    extra={
                        'user_id': keycloak_user_id,
                    },
                )
                return
            logger.error(
                'error_updating_litellm_key',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'user_id': keycloak_user_id,
                },
            )
        response.raise_for_status()

    @staticmethod
    async def _delete_user(
        client: httpx.AsyncClient,
        keycloak_user_id: str,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/user/delete', json={'user_ids': [keycloak_user_id]}
        )

        if not response.is_success:
            logger.error(
                'error_deleting_litellm_user',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'user_id': [keycloak_user_id],
                },
            )
        response.raise_for_status()

    @staticmethod
    async def _delete_team(
        client: httpx.AsyncClient,
        team_id: str,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/team/delete',
            json={'team_ids': [team_id]},
        )

        if not response.is_success:
            if response.status_code == 404:
                # Team doesn't exist, that's fine
                logger.info(
                    'Team already deleted or does not exist',
                    extra={'team_id': team_id},
                )
                return
            logger.error(
                'error_deleting_litellm_team',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'team_id': team_id,
                },
            )
        response.raise_for_status()
        logger.info(
            'LiteLlmManager:_delete_team:team_deleted',
            extra={'team_id': team_id},
        )

    @staticmethod
    async def _add_user_to_team(
        client: httpx.AsyncClient,
        keycloak_user_id: str,
        team_id: str,
        max_budget: float,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/team/member_add',
            json={
                'team_id': team_id,
                'member': {'user_id': keycloak_user_id, 'role': 'user'},
                'max_budget_in_team': max_budget,
            },
        )
        # Failed to add user to team - this is an unforseen error state...
        if not response.is_success:
            if (
                response.status_code == 400
                and 'already in team' in response.text.lower()
            ):
                logger.warning(
                    'user_already_in_team',
                    extra={
                        'user_id': keycloak_user_id,
                        'team_id': team_id,
                    },
                )
                return
            logger.error(
                'error_adding_litellm_user_to_team',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'user_id': [keycloak_user_id],
                    'team_id': [team_id],
                    'max_budget': max_budget,
                },
            )
        response.raise_for_status()

    @staticmethod
    async def _get_user_team_info(
        client: httpx.AsyncClient,
        keycloak_user_id: str,
        team_id: str,
    ) -> dict | None:
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        team_info = await LiteLlmManager._get_team(client, team_id)
        if not team_info:
            return None

        # Filter team_memberships based on team_id and keycloak_user_id
        user_membership = next(
            (
                membership
                for membership in team_info.get('team_memberships', [])
                if membership.get('user_id') == keycloak_user_id
                and membership.get('team_id') == team_id
            ),
            None,
        )

        return user_membership

    @staticmethod
    async def _update_user_in_team(
        client: httpx.AsyncClient,
        keycloak_user_id: str,
        team_id: str,
        max_budget: float,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/team/member_update',
            json={
                'team_id': team_id,
                'user_id': keycloak_user_id,
                'max_budget_in_team': max_budget,
            },
        )
        # Failed to update user in team - this is an unforseen error state...
        if not response.is_success:
            logger.error(
                'error_updating_litellm_user_in_team',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'user_id': [keycloak_user_id],
                    'team_id': [team_id],
                    'max_budget': max_budget,
                },
            )
        response.raise_for_status()

    @staticmethod
    async def _generate_key(
        client: httpx.AsyncClient,
        keycloak_user_id: str,
        team_id: str | None,
        key_alias: str | None,
        metadata: dict | None,
    ) -> str | None:
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        json_data: dict[str, Any] = {
            'user_id': keycloak_user_id,
            'models': [],
        }

        if team_id is not None:
            json_data['team_id'] = team_id

        if key_alias is not None:
            json_data['key_alias'] = key_alias

        if metadata is not None:
            json_data['metadata'] = metadata

        response = await client.post(
            f'{LITE_LLM_API_URL}/key/generate',
            json=json_data,
        )
        # Failed to generate user key for team - this is an unforseen error state...
        if not response.is_success:
            logger.error(
                'error_generate_user_team_key',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'user_id': keycloak_user_id,
                    'team_id': team_id,
                    'key_alias': key_alias,
                },
            )
        response.raise_for_status()
        response_json = response.json()
        key = response_json['key']
        logger.info(
            'LiteLlmManager:_lite_llm_generate_user_team_key:key_created',
            extra={
                'user_id': keycloak_user_id,
                'team_id': team_id,
                'key_alias': key_alias,
            },
        )
        return key

    @staticmethod
    async def verify_key(key: str, user_id: str) -> bool:
        """Verify that a key is valid in LiteLLM by making a lightweight API call.

        Args:
            key: The key to verify
            user_id: The user ID for logging purposes

        Returns:
            True if the key is verified as valid, False if verification fails or key is invalid.
            Returns False on network errors/timeouts to ensure we don't return potentially invalid keys.
        """
        if not (LITE_LLM_API_URL and key):
            return False

        try:
            async with httpx.AsyncClient(
                verify=httpx_verify_option(),
                timeout=BYOR_KEY_VERIFICATION_TIMEOUT,
            ) as client:
                # Make a lightweight request to verify the key
                # Using /v1/models endpoint as it's lightweight and requires authentication
                response = await client.get(
                    f'{LITE_LLM_API_URL}/v1/models',
                    headers={
                        'Authorization': f'Bearer {key}',
                    },
                )

                # Only 200 status code indicates valid key
                if response.status_code == 200:
                    logger.debug(
                        'BYOR key verification successful',
                        extra={'user_id': user_id},
                    )
                    return True

                # All other status codes (401, 403, 500, etc.) are treated as invalid
                # This includes authentication errors and server errors
                logger.warning(
                    'BYOR key verification failed - treating as invalid',
                    extra={
                        'user_id': user_id,
                        'status_code': response.status_code,
                        'key_prefix': key[:10] + '...' if len(key) > 10 else key,
                    },
                )
                return False

        except (httpx.TimeoutException, Exception) as e:
            # Any exception (timeout, network error, etc.) means we can't verify
            # Return False to trigger regeneration rather than returning potentially invalid key
            logger.warning(
                'BYOR key verification error - treating as invalid to ensure key validity',
                extra={
                    'user_id': user_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                },
            )
            return False

    @staticmethod
    async def _get_key_info(
        client: httpx.AsyncClient,
        org_id: str,
        keycloak_user_id: str,
    ) -> dict | None:
        from storage.user_store import UserStore

        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        user = await UserStore.get_user_by_id_async(keycloak_user_id)
        if not user:
            return {}

        org_member = None
        for om in user.org_members:
            if om.org_id == org_id:
                org_member = om
                break
        if not org_member or not org_member.llm_api_key:
            return {}
        response = await client.get(
            f'{LITE_LLM_API_URL}/key/info?key={org_member.llm_api_key}'
        )
        response.raise_for_status()
        response_json = response.json()
        key_info = response_json.get('info')
        if not key_info:
            return {}
        return {
            'key_max_budget': key_info.get('max_budget'),
            'key_spend': key_info.get('spend'),
        }

    @staticmethod
    async def _delete_key_by_alias(
        client: httpx.AsyncClient,
        key_alias: str,
    ):
        """Delete a key from LiteLLM by its alias.

        This is a best-effort operation that logs but does not raise on failure.
        """
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/key/delete',
            json={
                'key_aliases': [key_alias],
            },
        )
        if response.is_success:
            logger.info(
                'LiteLlmManager:_delete_key_by_alias:key_deleted',
                extra={'key_alias': key_alias},
            )
        elif response.status_code != 404:
            # Log non-404 errors but don't fail
            logger.warning(
                'error_deleting_key_by_alias',
                extra={
                    'key_alias': key_alias,
                    'status_code': response.status_code,
                    'text': response.text,
                },
            )

    @staticmethod
    async def _delete_key(
        client: httpx.AsyncClient,
        key_id: str,
        key_alias: str | None = None,
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/key/delete',
            json={
                'keys': [key_id],
            },
        )
        # Failed to delete key...
        if not response.is_success:
            if response.status_code == 404:
                # Key doesn't exist by key_id. If we have a key_alias,
                # try deleting by alias to clean up any orphaned alias.
                if key_alias:
                    await LiteLlmManager._delete_key_by_alias(client, key_alias)
                return
            logger.error(
                'error_deleting_key',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                },
            )
        response.raise_for_status()
        logger.info(
            'LiteLlmManager:_delete_key:key_deleted',
        )

    @staticmethod
    def with_http_client(
        internal_fn: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(internal_fn)
        async def wrapper(*args, **kwargs):
            async with httpx.AsyncClient(
                headers={'x-goog-api-key': LITE_LLM_API_KEY}
            ) as client:
                return await internal_fn(client, *args, **kwargs)

        return wrapper

    # Public methods with injected client
    create_team = staticmethod(with_http_client(_create_team))
    get_team = staticmethod(with_http_client(_get_team))
    update_team = staticmethod(with_http_client(_update_team))
    create_user = staticmethod(with_http_client(_create_user))
    get_user = staticmethod(with_http_client(_get_user))
    update_user = staticmethod(with_http_client(_update_user))
    delete_user = staticmethod(with_http_client(_delete_user))
    delete_team = staticmethod(with_http_client(_delete_team))
    add_user_to_team = staticmethod(with_http_client(_add_user_to_team))
    get_user_team_info = staticmethod(with_http_client(_get_user_team_info))
    update_user_in_team = staticmethod(with_http_client(_update_user_in_team))
    generate_key = staticmethod(with_http_client(_generate_key))
    get_key_info = staticmethod(with_http_client(_get_key_info))
    delete_key = staticmethod(with_http_client(_delete_key))
