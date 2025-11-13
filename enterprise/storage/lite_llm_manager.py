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


class LiteLlmManager:
    """Manage LiteLLM interactions."""

    @staticmethod
    async def create_entries(
        org_id: str,
        keycloak_user_id: str,
        oss_settings: Settings,
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
                    f'OpenHands Cloud - user {keycloak_user_id}',
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
            'SettingsStore:umigrate_lite_llm_entries:start',
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
                user_json = await LiteLlmManager._get_user(client, keycloak_user_id)
                if not user_json:
                    return None
                user_info = user_json['user_info']
                max_budget = user_info.get('max_budget', 0.0)
                if not max_budget:
                    # if max_budget is None, then we've already migrated the User
                    return None
                spend = user_info.get('spend', 0.0)
                credits = max(max_budget - spend, 0.0)

                await LiteLlmManager._create_team(
                    client, keycloak_user_id, org_id, credits
                )

                await LiteLlmManager._delete_user(client, keycloak_user_id)

                await LiteLlmManager._create_user(
                    client, keycloak_user_info.get('email'), keycloak_user_id
                )

                await LiteLlmManager._add_user_to_team(
                    client, keycloak_user_id, org_id, credits
                )

                key = await LiteLlmManager._generate_key(
                    client,
                    keycloak_user_id,
                    org_id,
                    f'OpenHands Cloud - user {keycloak_user_id}',
                    None,
                )

        user_settings.agent = 'CodeActAgent'
        # Use the model corresponding to the current user settings version
        user_settings.llm_model = get_default_litellm_model()
        user_settings.llm_api_key = SecretStr(key)
        user_settings.llm_base_url = LITE_LLM_API_URL
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
                },
            )

            # User failed to create in litellm - this is an unforseen error state...
            if not response.is_success:
                if response.status_code == 400 and 'already exists' in response.text:
                    # user already exists, just return
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
    ):
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return
        response = await client.post(
            f'{LITE_LLM_API_URL}/user/update',
            json={
                'user_id': keycloak_user_id,
                'metadata': {
                    'version': ORG_SETTINGS_VERSION,
                    'model': get_default_litellm_model(),
                },
            },
        )

        if not response.is_success:
            logger.error(
                'error_updating_litellm_user',
                extra={
                    'status_code': response.status_code,
                    'text': response.text,
                    'user_id': [keycloak_user_id],
                    'email': None,
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
    async def _get_key_info(
        client: httpx.AsyncClient,
        org_id: str,
        keycloak_user_id: str,
    ) -> dict | None:
        from storage.user_store import UserStore

        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            logger.warning('LiteLLM API configuration not found')
            return None
        user = UserStore.get_user_by_id(keycloak_user_id)
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
    async def _delete_key(
        client: httpx.AsyncClient,
        key_id: str,
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
        # Failed to key...
        if not response.is_success:
            if response.status_code == 404:
                # key doesn't exist, just return
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
    add_user_to_team = staticmethod(with_http_client(_add_user_to_team))
    get_user_team_info = staticmethod(with_http_client(_get_user_team_info))
    update_user_in_team = staticmethod(with_http_client(_update_user_in_team))
    generate_key = staticmethod(with_http_client(_generate_key))
    get_key_info = staticmethod(with_http_client(_get_key_info))
    delete_key = staticmethod(with_http_client(_delete_key))
