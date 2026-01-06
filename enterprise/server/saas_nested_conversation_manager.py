from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

import httpx
import socketio
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.constants import PERMITTED_CORS_ORIGINS, WEB_HOST
from server.utils.conversation_callback_utils import (
    process_event,
    update_conversation_metadata,
)
from sqlalchemy import orm
from storage.api_key_store import ApiKeyStore
from storage.database import session_maker
from storage.stored_conversation_metadata import StoredConversationMetadata

from openhands.controller.agent import Agent
from openhands.core.config import LLMConfig, OpenHandsConfig
from openhands.core.config.mcp_config import MCPConfig, MCPSHTTPServerConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import MessageAction
from openhands.events.event_store import EventStore
from openhands.events.serialization.event import event_to_dict
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
    ProviderToken,
)
from openhands.runtime.impl.remote.remote_runtime import RemoteRuntime
from openhands.runtime.plugins.vscode import VSCodeRequirement
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.config.server_config import ServerConfig
from openhands.server.constants import ROOM_KEY
from openhands.server.conversation_manager.conversation_manager import (
    ConversationManager,
)
from openhands.server.data_models.agent_loop_info import AgentLoopInfo
from openhands.server.monitoring import MonitoringListener
from openhands.server.session import Session
from openhands.server.session.conversation import ServerConversation
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.conversation_metadata import ConversationMetadata
from openhands.storage.data_models.conversation_status import ConversationStatus
from openhands.storage.data_models.settings import Settings
from openhands.storage.files import FileStore
from openhands.storage.locations import (
    get_conversation_event_filename,
    get_conversation_events_dir,
)
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.http_session import httpx_verify_option
from openhands.utils.import_utils import get_impl
from openhands.utils.shutdown_listener import should_continue
from openhands.utils.utils import create_registry_and_conversation_stats

# Pattern for accessing runtime pods externally
RUNTIME_URL_PATTERN = os.getenv(
    'RUNTIME_URL_PATTERN', 'https://{runtime_id}.prod-runtime.all-hands.dev'
)
RUNTIME_ROUTING_MODE = os.getenv('RUNTIME_ROUTING_MODE', 'subdomain').lower()

# Pattern for base URL for the runtime
RUNTIME_CONVERSATION_URL = RUNTIME_URL_PATTERN + (
    '/runtime/api/conversations/{conversation_id}'
    if RUNTIME_ROUTING_MODE == 'path'
    else '/api/conversations/{conversation_id}'
)

RUNTIME_USERNAME = os.getenv('RUNTIME_USERNAME')

SU_TO_USER = os.getenv('SU_TO_USER', 'false')
truthy = {'1', 'true', 't', 'yes', 'y', 'on'}
SU_TO_USER = str(SU_TO_USER.lower() in truthy).lower()

DISABLE_VSCODE_PLUGIN = os.getenv('DISABLE_VSCODE_PLUGIN', 'false').lower() == 'true'

# Time in seconds before a Redis entry is considered expired if not refreshed
_REDIS_ENTRY_TIMEOUT_SECONDS = 300

# Time in seconds between pulls
_POLLING_INTERVAL = 10

# Timeout for http operations
_HTTP_TIMEOUT = 15


class EventRetrieval(Enum):
    """Determine mode for getting events out of the nested runtime back into the main app."""

    WEBHOOK_PUSH = 'WEBHOOK_PUSH'
    POLLING = 'POLLING'
    NONE = 'NONE'


@dataclass
class SaasNestedConversationManager(ConversationManager):
    """Conversation manager where the agent loops exist inside the remote containers."""

    sio: socketio.AsyncServer
    config: OpenHandsConfig
    server_config: ServerConfig
    file_store: FileStore
    event_retrieval: EventRetrieval
    _conversation_store_class: type[ConversationStore] | None = None
    _event_polling_task: asyncio.Task | None = None
    _runtime_container_image: str | None = None

    async def __aenter__(self):
        if self.event_retrieval == EventRetrieval.POLLING:
            self._event_polling_task = asyncio.create_task(self._poll_events())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._event_polling_task:
            self._event_polling_task.cancel()
            self._event_polling_task = None

    async def attach_to_conversation(
        self, sid: str, user_id: str | None = None
    ) -> ServerConversation | None:
        # Not supported - clients should connect directly to the nested server!
        raise ValueError('unsupported_operation')

    async def detach_from_conversation(self, conversation: ServerConversation):
        # Not supported - clients should connect directly to the nested server!
        raise ValueError('unsupported_operation')

    async def join_conversation(
        self,
        sid: str,
        connection_id: str,
        settings: Settings,
        user_id: str | None,
    ) -> AgentLoopInfo:
        # Not supported - clients should connect directly to the nested server!
        raise ValueError('unsupported_operation')

    def get_agent_session(self, sid: str):
        raise ValueError('unsupported_operation')

    async def get_running_agent_loops(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> set[str]:
        """
        Get the running agent loops directly from the remote runtime.
        """
        conversation_ids = await self._get_all_running_conversation_ids()

        if filter_to_sids is not None:
            conversation_ids = {
                conversation_id
                for conversation_id in conversation_ids
                if conversation_id in filter_to_sids
            }

        if user_id:
            user_conversation_ids = await call_sync_from_async(
                self._get_recent_conversation_ids_for_user, user_id
            )
            conversation_ids = conversation_ids.intersection(user_conversation_ids)

        return conversation_ids

    async def is_agent_loop_running(self, sid: str) -> bool:
        """Check if an agent loop is running for the given session ID."""
        runtime = await self._get_runtime(sid)
        if runtime is None:
            return False
        result = runtime.get('status') == 'running'
        return result

    async def get_connections(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> dict[str, str]:
        # We don't monitor connections outside the nested server, though we could introduce an API for this.
        results: dict[str, str] = {}
        return results

    async def maybe_start_agent_loop(
        self,
        sid: str,
        settings: Settings,
        user_id: str,  # type: ignore[override]
        initial_user_msg: MessageAction | None = None,
        replay_json: str | None = None,
    ) -> AgentLoopInfo:
        # First we check redis to see if we are already starting - or the runtime will tell us the session is stopped
        redis = self._get_redis_client()
        key = self._get_redis_conversation_key(user_id, sid)
        starting = await redis.get(key)

        runtime = await self._get_runtime(sid)

        nested_url = None
        session_api_key = None
        status = ConversationStatus.STOPPED
        event_store = EventStore(sid, self.file_store, user_id)
        if runtime:
            nested_url = self._get_nested_url_for_runtime(runtime['runtime_id'], sid)
            session_api_key = runtime.get('session_api_key')
            status_str = (runtime.get('status') or 'stopped').upper()
            if status_str in ConversationStatus:
                status = ConversationStatus[status_str]
        if status is ConversationStatus.STOPPED and starting:
            status = ConversationStatus.STARTING

        if status is ConversationStatus.STOPPED:
            # Mark the agentloop as starting in redis
            await redis.set(key, 1, ex=_REDIS_ENTRY_TIMEOUT_SECONDS)

            # Start the agent loop in the background
            asyncio.create_task(
                self._start_agent_loop(
                    sid, settings, user_id, initial_user_msg, replay_json
                )
            )

        return AgentLoopInfo(
            conversation_id=sid,
            url=nested_url,
            session_api_key=session_api_key,
            event_store=event_store,
            status=status,
        )

    async def _refresh_provider_tokens_after_runtime_init(
        self, settings: Settings, sid: str, user_id: str | None = None
    ) -> Settings:
        """Refresh provider tokens after runtime initialization.

        During runtime initialization, tokens may be refreshed by Runtime.__init__().
        This method retrieves the fresh tokens from the database and creates a new
        settings object with updated tokens to avoid sending stale tokens to the
        nested runtime.

        The method handles two scenarios:
        1. ProviderToken has user_id (IDP user ID, e.g., GitLab user ID)
           → Uses get_idp_token_from_idp_user_id()
        2. ProviderToken has no user_id but Keycloak user_id is available
           → Uses load_offline_token() + get_idp_token_from_offline_token()

        Args:
            settings: The conversation settings that may contain provider tokens
            sid: The session ID for logging purposes
            user_id: The Keycloak user ID (optional, used as fallback when
                     ProviderToken.user_id is not available)

        Returns:
            Updated settings with fresh provider tokens, or original settings
            if no update is needed
        """
        if not isinstance(settings, ConversationInitData):
            return settings

        if not settings.git_provider_tokens:
            return settings

        token_manager = TokenManager()
        updated_tokens = {}
        tokens_refreshed = 0
        tokens_failed = 0

        for provider_type, provider_token in settings.git_provider_tokens.items():
            fresh_token = None

            try:
                if provider_token.user_id:
                    # Case 1: We have IDP user ID (e.g., GitLab user ID '32546706')
                    # Get the token that was just refreshed during runtime initialization
                    fresh_token = await token_manager.get_idp_token_from_idp_user_id(
                        provider_token.user_id, provider_type
                    )
                elif user_id:
                    # Case 2: We have Keycloak user ID but no IDP user ID
                    # This happens in web UI flow where ProviderToken.user_id is None
                    offline_token = await token_manager.load_offline_token(user_id)
                    if offline_token:
                        fresh_token = (
                            await token_manager.get_idp_token_from_offline_token(
                                offline_token, provider_type
                            )
                        )

                if fresh_token:
                    updated_tokens[provider_type] = ProviderToken(
                        token=SecretStr(fresh_token),
                        user_id=provider_token.user_id,
                        host=provider_token.host,
                    )
                    tokens_refreshed += 1
                else:
                    # Keep original token if we couldn't get a fresh one
                    updated_tokens[provider_type] = provider_token

            except Exception as e:
                # If refresh fails, use original token to prevent conversation startup failure
                logger.warning(
                    f'Failed to refresh {provider_type.value} token: {e}',
                    extra={'session_id': sid, 'provider': provider_type.value},
                    exc_info=True,
                )
                updated_tokens[provider_type] = provider_token
                tokens_failed += 1

        # Create new ConversationInitData with updated tokens
        # We cannot modify the frozen field directly, so we create a new object
        updated_settings = settings.model_copy(
            update={'git_provider_tokens': MappingProxyType(updated_tokens)}
        )

        logger.info(
            'Updated provider tokens after runtime creation',
            extra={
                'session_id': sid,
                'providers': [p.value for p in updated_tokens.keys()],
                'refreshed': tokens_refreshed,
                'failed': tokens_failed,
            },
        )
        return updated_settings

    async def _start_agent_loop(
        self, sid, settings, user_id, initial_user_msg=None, replay_json=None
    ):
        try:
            logger.info(f'starting_agent_loop:{sid}', extra={'session_id': sid})
            await self.ensure_num_conversations_below_limit(sid, user_id)
            provider_handler = self._get_provider_handler(settings)
            runtime = await self._create_runtime(
                sid, user_id, settings, provider_handler
            )
            await runtime.connect()

            if not self._runtime_container_image:
                self._runtime_container_image = getattr(
                    runtime,
                    'container_image',
                    self.config.sandbox.runtime_container_image,
                )

            session_api_key = runtime.session.headers['X-Session-API-Key']

            # Update provider tokens with fresh ones after runtime creation
            settings = await self._refresh_provider_tokens_after_runtime_init(
                settings, sid, user_id
            )

            await self._start_conversation(
                sid,
                user_id,
                settings,
                initial_user_msg,
                replay_json,
                runtime.runtime_url,
                session_api_key,
            )
        finally:
            # remove the starting entry from redis
            redis = self._get_redis_client()
            key = self._get_redis_conversation_key(user_id, sid)
            await redis.delete(key)

    async def _start_conversation(
        self,
        sid: str,
        user_id: str,
        settings: Settings,
        initial_user_msg: MessageAction | None,
        replay_json: str | None,
        api_url: str,
        session_api_key: str,
    ):
        logger.info('starting_nested_conversation', extra={'sid': sid})
        async with httpx.AsyncClient(
            verify=httpx_verify_option(),
            headers={
                'X-Session-API-Key': session_api_key,
            },
        ) as client:
            await self._setup_nested_settings(client, api_url, settings)
            await self._setup_provider_tokens(client, api_url, settings)
            await self._setup_custom_secrets(client, api_url, settings.custom_secrets)  # type: ignore
            await self._setup_experiment_config(client, api_url, sid, user_id)
            await self._create_nested_conversation(
                client, api_url, sid, user_id, settings, initial_user_msg, replay_json
            )
            await self._wait_for_conversation_ready(client, api_url, sid)

    async def _setup_experiment_config(
        self, client: httpx.AsyncClient, api_url: str, sid: str, user_id: str
    ):
        # Prevent circular import
        from openhands.experiments.experiment_manager import (
            ExperimentConfig,
            ExperimentManagerImpl,
        )

        config: OpenHandsConfig = ExperimentManagerImpl.run_config_variant_test(
            user_id, sid, self.config
        )

        experiment_config = ExperimentConfig(
            config={
                'system_prompt_filename': config.get_agent_config(
                    config.default_agent
                ).system_prompt_filename
            }
        )

        response = await client.post(
            f'{api_url}/api/conversations/{sid}/exp-config',
            json=experiment_config.model_dump(),
        )
        response.raise_for_status()

    async def _setup_nested_settings(
        self, client: httpx.AsyncClient, api_url: str, settings: Settings
    ) -> None:
        """Setup the settings for the nested conversation."""
        settings_json = settings.model_dump(context={'expose_secrets': True})
        settings_json.pop('custom_secrets', None)
        settings_json.pop('git_provider_tokens', None)
        if settings_json.get('git_provider'):
            settings_json['git_provider'] = settings_json['git_provider'].value
        settings_json.pop('secrets_store', None)
        response = await client.post(f'{api_url}/api/settings', json=settings_json)
        response.raise_for_status()

    async def _setup_provider_tokens(
        self, client: httpx.AsyncClient, api_url: str, settings: Settings
    ):
        """Setup provider tokens for the nested conversation.

        Note: Token validation happens in the nested runtime. If tokens are revoked,
        the nested runtime will return 401. The caller should handle token refresh
        and retry if needed.
        """
        provider_handler = self._get_provider_handler(settings)
        provider_tokens = provider_handler.provider_tokens
        if provider_tokens:
            provider_tokens_json = {
                k.value: {
                    'token': v.token.get_secret_value(),
                    'user_id': v.user_id,
                    'host': v.host,
                }
                for k, v in provider_tokens.items()
                if v.token
            }
            response = await client.post(
                f'{api_url}/api/add-git-providers',
                json={
                    'provider_tokens': provider_tokens_json,
                },
            )
            response.raise_for_status()

    async def _setup_custom_secrets(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        custom_secrets: MappingProxyType[str, Any] | None,
    ):
        """Setup custom secrets for the nested conversation.

        Note: When resuming conversations, secrets may already exist in the runtime.
        We check for specific duplicate error messages to handle this case gracefully.
        """
        if custom_secrets:
            for key, secret in custom_secrets.items():
                try:
                    response = await client.post(
                        f'{api_url}/api/secrets',
                        json={
                            'name': key,
                            'description': secret.description,
                            'value': secret.secret.get_secret_value(),
                        },
                    )
                    response.raise_for_status()
                    logger.debug(f'Successfully created secret: {key}')
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 400:
                        # Only ignore if it's actually a duplicate error
                        try:
                            error_data = e.response.json()
                            error_msg = error_data.get('message', '')
                            # The API returns: "Secret {secret_name} already exists"
                            if 'already exists' in error_msg:
                                logger.info(
                                    f'Secret "{key}" already exists, continuing - ignoring duplicate',
                                    extra={'api_url': api_url},
                                )
                                continue
                        except (KeyError, ValueError, TypeError):
                            pass  # If we can't parse JSON, fall through to re-raise
                    # Re-raise all other errors (including non-duplicate 400s)
                    logger.error(
                        f'Failed to setup secret "{key}": HTTP {e.response.status_code}',
                        extra={
                            'api_url': api_url,
                            'response_text': e.response.text[:200],
                        },
                    )
                    raise

    def _get_mcp_config(self, user_id: str) -> MCPConfig | None:
        api_key_store = ApiKeyStore.get_instance()
        mcp_api_key = api_key_store.retrieve_mcp_api_key(user_id)
        if not mcp_api_key:
            mcp_api_key = api_key_store.create_api_key(user_id, 'MCP_API_KEY', None)
        if not mcp_api_key:
            return None
        web_host = os.environ.get('WEB_HOST', 'app.all-hands.dev')
        shttp_servers = [
            MCPSHTTPServerConfig(url=f'https://{web_host}/mcp/mcp', api_key=mcp_api_key)
        ]
        return MCPConfig(shttp_servers=shttp_servers)

    async def _create_nested_conversation(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        sid: str,
        user_id: str,
        settings: Settings,
        initial_user_msg: MessageAction | None,
        replay_json: str | None,
    ):
        """Create the nested conversation."""
        init_conversation: dict[str, Any] = {
            'initial_user_msg': initial_user_msg.content if initial_user_msg else None,
            'image_urls': [],
            'replay_json': replay_json,
            'conversation_id': sid,
        }

        mcp_config = self._get_mcp_config(user_id)
        if mcp_config:
            # Merge with any MCP config from settings
            if settings.mcp_config:
                mcp_config = mcp_config.merge(settings.mcp_config)
            # Check again since theoretically merge could return None.
            if mcp_config:
                init_conversation['mcp_config'] = mcp_config.model_dump()

        if isinstance(settings, ConversationInitData):
            init_conversation['repository'] = settings.selected_repository
            init_conversation['selected_branch'] = settings.selected_branch
            init_conversation['git_provider'] = (
                settings.git_provider.value if settings.git_provider else None
            )
            init_conversation['conversation_instructions'] = (
                settings.conversation_instructions
            )

        response = await client.post(
            f'{api_url}/api/conversations', json=init_conversation
        )
        logger.info(f'_start_agent_loop:{response.status_code}:{response.json()}')
        response.raise_for_status()

    async def _wait_for_conversation_ready(
        self, client: httpx.AsyncClient, api_url: str, sid: str
    ):
        """Wait for the conversation to be ready by checking the events endpoint."""
        # TODO: Find out why /api/conversations/{sid} returns RUNNING when events are not available
        for _ in range(5):
            try:
                logger.info('checking_events_endpoint_running', extra={'sid': sid})
                response = await client.get(f'{api_url}/api/conversations/{sid}/events')
                if response.is_success:
                    logger.info('events_endpoint_is_running', extra={'sid': sid})
                    break
            except Exception:
                logger.warning('events_endpoint_not_ready', extra={'sid': sid})
            await asyncio.sleep(5)

    async def send_to_event_stream(self, connection_id: str, data: dict):
        # Not supported - clients should connect directly to the nested server!
        raise ValueError('unsupported_operation')

    async def request_llm_completion(
        self,
        sid: str,
        service_id: str,
        llm_config: LLMConfig,
        messages: list[dict[str, str]],
    ) -> str:
        # Not supported - clients should connect directly to the nested server!
        raise ValueError('unsupported_operation')

    async def send_event_to_conversation(self, sid: str, data: dict):
        runtime = await self._get_runtime(sid)
        if runtime is None:
            raise ValueError(f'no_such_conversation:{sid}')
        nested_url = self._get_nested_url_for_runtime(runtime['runtime_id'], sid)
        async with httpx.AsyncClient(
            verify=httpx_verify_option(),
            headers={
                'X-Session-API-Key': runtime['session_api_key'],
            },
        ) as client:
            response = await client.post(f'{nested_url}/events', json=data)
            response.raise_for_status()

    async def disconnect_from_session(self, connection_id: str):
        # Not supported - clients should connect directly to the nested server!
        raise ValueError('unsupported_operation')

    async def close_session(self, sid: str):
        logger.info('close_session', extra={'sid': sid})
        runtime = await self._get_runtime(sid)
        if runtime is None:
            logger.info('no_session_to_close', extra={'sid': sid})
            return
        async with self._httpx_client() as client:
            response = await client.post(
                f'{self.remote_runtime_api_url}/pause',
                json={'runtime_id': runtime['runtime_id']},
            )
            if not response.is_success:
                logger.info(
                    'failed_to_close_session',
                    {
                        'sid': sid,
                        'status_code': response.status_code,
                        'detail': (response.content or b'').decode(),
                    },
                )

    def _get_user_id_from_conversation(self, conversation_id: str) -> str:
        """
        Get user_id from conversation_id.
        """

        with session_maker() as session:
            conversation_metadata = (
                session.query(StoredConversationMetadata)
                .filter(StoredConversationMetadata.conversation_id == conversation_id)
                .first()
            )

            if not conversation_metadata:
                raise ValueError(f'No conversation found {conversation_id}')

            return conversation_metadata.user_id

    async def _get_runtime_status_from_nested_runtime(
        self, session_api_key: Any | None, nested_url: str, conversation_id: str
    ) -> RuntimeStatus | None:
        """Get runtime status from the nested runtime via API call.

        Args:
            session_api_key: The session API key for authentication
            nested_url: The base URL of the nested runtime
            conversation_id: The conversation ID for logging purposes

        Returns:
            The runtime status if available, None otherwise
        """
        try:
            if not session_api_key:
                return None

            async with httpx.AsyncClient(
                verify=httpx_verify_option(),
                headers={
                    'X-Session-API-Key': session_api_key,
                },
            ) as client:
                # Query the nested runtime for conversation info
                response = await client.get(nested_url)
                if response.status_code == 200:
                    conversation_data = response.json()
                    runtime_status_str = conversation_data.get('runtime_status')
                    if runtime_status_str:
                        # Convert string back to RuntimeStatus enum
                        return RuntimeStatus(runtime_status_str)
                else:
                    logger.debug(
                        f'Failed to get conversation info for {conversation_id}: {response.status_code}'
                    )
        except ValueError:
            logger.debug(f'Invalid runtime status value: {runtime_status_str}')
        except Exception as e:
            logger.debug(f'Could not get runtime status for {conversation_id}: {e}')

        return None

    async def get_agent_loop_info(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> list[AgentLoopInfo]:
        if filter_to_sids is not None and not filter_to_sids:
            return []

        results = []
        conversation_ids = set()

        # Get starting agent loops from redis...
        if user_id:
            pattern = self._get_redis_conversation_key(user_id, '*')
        else:
            pattern = self._get_redis_conversation_key('*', '*')
        redis = self._get_redis_client()
        async for key in redis.scan_iter(pattern):
            conversation_user_id, conversation_id = key.decode().split(':')[1:]
            conversation_ids.add(conversation_id)
            if filter_to_sids is None or conversation_id in filter_to_sids:
                results.append(
                    AgentLoopInfo(
                        conversation_id=conversation_id,
                        url=None,
                        session_api_key=None,
                        event_store=EventStore(
                            conversation_id, self.file_store, conversation_user_id
                        ),
                        status=ConversationStatus.STARTING,
                    )
                )

        # Get running agent loops from runtime api
        if filter_to_sids and len(filter_to_sids) == 1:
            runtimes = []
            runtime = await self._get_runtime(next(iter(filter_to_sids)))
            if runtime:
                runtimes.append(runtime)
        else:
            runtimes = await self._get_runtimes()
        for runtime in runtimes:
            conversation_id = runtime['session_id']
            if conversation_id in conversation_ids:
                continue
            if filter_to_sids is not None and conversation_id not in filter_to_sids:
                continue

            user_id_for_convo = user_id
            if not user_id_for_convo:
                try:
                    user_id_for_convo = await call_sync_from_async(
                        self._get_user_id_from_conversation, conversation_id
                    )
                except Exception:
                    continue

            nested_url = self._get_nested_url_for_runtime(
                runtime['runtime_id'], conversation_id
            )
            session_api_key = runtime.get('session_api_key')

            # Get runtime status from nested runtime
            runtime_status = await self._get_runtime_status_from_nested_runtime(
                session_api_key, nested_url, conversation_id
            )

            agent_loop_info = AgentLoopInfo(
                conversation_id=conversation_id,
                url=nested_url,
                session_api_key=session_api_key,
                event_store=EventStore(
                    sid=conversation_id,
                    file_store=self.file_store,
                    user_id=user_id_for_convo,
                ),
                status=self._parse_status(runtime),
                runtime_status=runtime_status,
            )
            results.append(agent_loop_info)

        return results

    @classmethod
    def get_instance(
        cls,
        sio: socketio.AsyncServer,
        config: OpenHandsConfig,
        file_store: FileStore,
        server_config: ServerConfig,
        monitoring_listener: MonitoringListener,
    ) -> ConversationManager:
        if 'localhost' in WEB_HOST:
            event_retrieval = EventRetrieval.POLLING
        else:
            event_retrieval = EventRetrieval.WEBHOOK_PUSH
        return SaasNestedConversationManager(
            sio=sio,
            config=config,
            server_config=server_config,
            file_store=file_store,
            event_retrieval=event_retrieval,
        )

    @property
    def remote_runtime_api_url(self):
        return self.config.sandbox.remote_runtime_api_url

    async def _get_conversation_store(self, user_id: str | None) -> ConversationStore:
        conversation_store_class = self._conversation_store_class
        if not conversation_store_class:
            self._conversation_store_class = conversation_store_class = get_impl(
                ConversationStore,  # type: ignore
                self.server_config.conversation_store_class,
            )
        store = await conversation_store_class.get_instance(self.config, user_id)  # type: ignore
        return store

    async def ensure_num_conversations_below_limit(self, sid: str, user_id: str):
        response_ids = await self.get_running_agent_loops(user_id)
        if len(response_ids) >= self.config.max_concurrent_conversations:
            logger.info(
                f'too_many_sessions_for:{user_id or ""}',
                extra={'session_id': sid, 'user_id': user_id},
            )
            # Get the conversations sorted (oldest first)
            conversation_store = await self._get_conversation_store(user_id)
            conversations = await conversation_store.get_all_metadata(response_ids)
            conversations.sort(key=_last_updated_at_key, reverse=True)

            while len(conversations) >= self.config.max_concurrent_conversations:
                oldest_conversation_id = conversations.pop().conversation_id
                logger.debug(
                    f'closing_from_too_many_sessions:{user_id or ""}:{oldest_conversation_id}',
                    extra={'session_id': oldest_conversation_id, 'user_id': user_id},
                )
                # Send status message to client and close session.
                status_update_dict = {
                    'status_update': True,
                    'type': 'error',
                    'id': 'AGENT_ERROR$TOO_MANY_CONVERSATIONS',
                    'message': 'Too many conversations at once. If you are still using this one, try reactivating it by prompting the agent to continue',
                }
                await self.sio.emit(
                    'oh_event',
                    status_update_dict,
                    to=ROOM_KEY.format(sid=oldest_conversation_id),
                )
                await self.close_session(oldest_conversation_id)

    def _get_provider_handler(self, settings: Settings):
        provider_tokens = None
        if isinstance(settings, ConversationInitData):
            provider_tokens = settings.git_provider_tokens
        provider_handler = ProviderHandler(
            provider_tokens=provider_tokens
            or cast(PROVIDER_TOKEN_TYPE, MappingProxyType({}))
        )
        return provider_handler

    async def _create_runtime(
        self,
        sid: str,
        user_id: str,
        settings: Settings,
        provider_handler: ProviderHandler,
    ):
        llm_registry, conversation_stats, config = (
            create_registry_and_conversation_stats(self.config, sid, user_id, settings)
        )

        # This session is created here only because it is the easiest way to get a runtime, which
        # is the easiest way to create the needed docker container
        session = Session(
            sid=sid,
            llm_registry=llm_registry,
            conversation_stats=conversation_stats,
            file_store=self.file_store,
            config=self.config,
            sio=self.sio,
            user_id=user_id,
        )
        llm_registry.retry_listner = session._notify_on_llm_retry
        agent_cls = settings.agent or self.config.default_agent
        agent_config = self.config.get_agent_config(agent_cls)
        agent = Agent.get_cls(agent_cls)(agent_config, llm_registry)

        config = self.config.model_copy(deep=True)
        env_vars = config.sandbox.runtime_startup_env_vars
        env_vars['CONVERSATION_MANAGER_CLASS'] = (
            'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager'
        )
        env_vars['LOG_JSON'] = '1'
        env_vars['SERVE_FRONTEND'] = '0'
        env_vars['RUNTIME'] = 'local'
        # TODO: In the long term we may come up with a more secure strategy for user management within the nested runtime.
        env_vars['USER'] = (
            RUNTIME_USERNAME
            if RUNTIME_USERNAME
            else ('openhands' if config.run_as_openhands else 'root')
        )
        env_vars['PERMITTED_CORS_ORIGINS'] = ','.join(PERMITTED_CORS_ORIGINS)
        env_vars['port'] = '60000'
        # TODO: These values are static in the runtime-api project, but do not get copied into the runtime ENV
        env_vars['VSCODE_PORT'] = '60001'
        env_vars['WORK_PORT_1'] = '12000'
        env_vars['WORK_PORT_2'] = '12001'
        # We need to be able to specify the nested conversation id within the nested runtime
        env_vars['ALLOW_SET_CONVERSATION_ID'] = '1'
        env_vars['FILE_STORE_PATH'] = '/workspace/.openhands/file_store'
        env_vars['WORKSPACE_BASE'] = '/workspace/project'
        env_vars['WORKSPACE_MOUNT_PATH_IN_SANDBOX'] = '/workspace/project'
        env_vars['SANDBOX_CLOSE_DELAY'] = '0'
        env_vars['SKIP_DEPENDENCY_CHECK'] = '1'
        env_vars['INITIAL_NUM_WARM_SERVERS'] = '1'
        env_vars['INIT_GIT_IN_EMPTY_WORKSPACE'] = '1'
        env_vars['ENABLE_V1'] = '0'
        env_vars['SU_TO_USER'] = SU_TO_USER
        env_vars['DISABLE_VSCODE_PLUGIN'] = str(DISABLE_VSCODE_PLUGIN).lower()
        env_vars['BROWSERGYM_DOWNLOAD_DIR'] = '/workspace/.downloads/'
        env_vars['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/playwright-browsers'

        # We need this for LLM traces tracking to identify the source of the LLM calls
        env_vars['WEB_HOST'] = WEB_HOST
        if self.event_retrieval == EventRetrieval.WEBHOOK_PUSH:
            # If we are retrieving events using push, we tell the nested runtime about the webhook.
            # The nested runtime will automatically authenticate using the SESSION_API_KEY
            env_vars['FILE_STORE_WEB_HOOK_URL'] = (
                f'{PERMITTED_CORS_ORIGINS[0]}/event-webhook/batch'
            )
            # Enable batched webhook mode for better performance
            env_vars['FILE_STORE_WEB_HOOK_BATCH'] = '1'

        if self._runtime_container_image:
            config.sandbox.runtime_container_image = self._runtime_container_image

        plugins = [
            plugin
            for plugin in agent.sandbox_plugins
            if not (DISABLE_VSCODE_PLUGIN and isinstance(plugin, VSCodeRequirement))
        ]
        logger.info(f'Loaded plugins for runtime {sid}: {plugins}')

        runtime = RemoteRuntime(
            config=config,
            event_stream=None,  # type: ignore[arg-type]
            sid=sid,
            plugins=plugins,
            # env_vars=env_vars,
            # status_callback: Callable[..., None] | None = None,
            attach_to_existing=False,
            headless_mode=False,
            user_id=user_id,
            # git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
            main_module='openhands.server',
            llm_registry=llm_registry,
        )

        # TODO: This is a hack. The setup_initial_env method directly calls the methods on the action
        # execution server, even though there are not any variables to set. In the nested env, there
        # is currently no direct access to the action execution server, so we should either add a
        # check and not invoke the endpoint if there are no variables, or find a way to access the
        # action execution server directly (e.g.: Merge the action execution server with the app
        # server for local runtimes)
        runtime.setup_initial_env = lambda: None  # type:ignore

        return runtime

    @contextlib.asynccontextmanager
    async def _httpx_client(self):
        async with httpx.AsyncClient(
            verify=httpx_verify_option(),
            headers={'X-API-Key': self.config.sandbox.api_key or ''},
            timeout=_HTTP_TIMEOUT,
        ) as client:
            yield client

    async def _get_runtimes(self) -> list[dict]:
        async with self._httpx_client() as client:
            response = await client.get(f'{self.remote_runtime_api_url}/list')
            response_json = response.json()
            runtimes = response_json['runtimes']
            return runtimes

    async def _get_all_running_conversation_ids(self) -> set[str]:
        runtimes = await self._get_runtimes()
        conversation_ids = {
            runtime['session_id']
            for runtime in runtimes
            if runtime.get('status') == 'running'
        }
        return conversation_ids

    def _get_recent_conversation_ids_for_user(self, user_id: str) -> set[str]:
        with session_maker() as session:
            # Only include conversations updated in the past week
            one_week_ago = datetime.now(UTC) - timedelta(days=7)
            query = session.query(StoredConversationMetadata.conversation_id).filter(
                StoredConversationMetadata.user_id == user_id,
                StoredConversationMetadata.last_updated_at >= one_week_ago,
            )
            user_conversation_ids = set(query)
            return user_conversation_ids

    async def _get_runtime(self, sid: str) -> dict | None:
        async with self._httpx_client() as client:
            response = await client.get(f'{self.remote_runtime_api_url}/sessions/{sid}')
            if not response.is_success:
                return None
            response_json = response.json()

            # Hack: This endpoint doesn't return the session_id
            response_json['session_id'] = sid

            return response_json

    def _parse_status(self, runtime: dict):
        # status is one of running, stoppped, paused, error, starting
        status = (runtime.get('status') or '').upper()
        if status == 'PAUSED':
            return ConversationStatus.STOPPED
        elif status == 'STOPPED':
            return ConversationStatus.ARCHIVED
        if status in ConversationStatus:
            return ConversationStatus[status]
        return ConversationStatus.STOPPED

    def _get_nested_url_for_runtime(self, runtime_id: str, conversation_id: str):
        return RUNTIME_CONVERSATION_URL.format(
            runtime_id=runtime_id, conversation_id=conversation_id
        )

    def _get_redis_client(self):
        return getattr(self.sio.manager, 'redis', None)

    def _get_redis_conversation_key(self, user_id: str, conversation_id: str):
        return f'ohcnv:{user_id}:{conversation_id}'

    async def _poll_events(self):
        """Poll events in nested runtimes. This is primarily used in debug / single server environments"""
        while should_continue():
            try:
                await asyncio.sleep(_POLLING_INTERVAL)
                agent_loop_infos = await self.get_agent_loop_info()

                with session_maker() as session:
                    for agent_loop_info in agent_loop_infos:
                        if agent_loop_info.status != ConversationStatus.RUNNING:
                            continue
                        try:
                            await self._poll_agent_loop_events(agent_loop_info, session)
                        except Exception as e:
                            logger.exception(f'error_polling_events:{str(e)}')
            except Exception as e:
                try:
                    asyncio.get_running_loop()
                    logger.exception(f'error_polling_events:{str(e)}')
                except RuntimeError:
                    # Loop has been shut down, exit gracefully
                    return

    async def _poll_agent_loop_events(
        self, agent_loop_info: AgentLoopInfo, session: orm.Session
    ):
        """This method is typically only run in localhost, where the webhook callbacks from the remote runtime are unavailable"""
        if agent_loop_info.status != ConversationStatus.RUNNING:
            return
        conversation_id = agent_loop_info.conversation_id
        conversation_metadata = (
            session.query(StoredConversationMetadata)
            .filter(StoredConversationMetadata.conversation_id == conversation_id)
            .first()
        )
        if conversation_metadata is None:
            # Conversation is running in different server
            return

        user_id = conversation_metadata.user_id

        # Get the id of the next event which is not present
        events_dir = get_conversation_events_dir(
            agent_loop_info.conversation_id, user_id
        )
        try:
            event_file_names = self.file_store.list(events_dir)
        except FileNotFoundError:
            event_file_names = []
        start_id = (
            max(
                (
                    _get_id_from_filename(event_file_name)
                    for event_file_name in event_file_names
                ),
                default=-1,
            )
            + 1
        )

        # Copy over any missing events and update the conversation metadata
        last_updated_at = conversation_metadata.last_updated_at
        if agent_loop_info.event_store:
            for event in agent_loop_info.event_store.search_events(start_id=start_id):
                # What would the handling be if no event.timestamp? Can that happen?
                if event.timestamp:
                    timestamp = datetime.fromisoformat(event.timestamp)
                    last_updated_at = max(last_updated_at, timestamp)
                contents = json.dumps(event_to_dict(event))
                path = get_conversation_event_filename(
                    conversation_id, event.id, user_id
                )
                self.file_store.write(path, contents)

                # Process the event using shared logic from event_webhook
                subpath = f'events/{event.id}.json'
                await process_event(
                    user_id, conversation_id, subpath, event_to_dict(event)
                )

        # Update conversation metadata using shared logic
        metadata_content = {
            'last_updated_at': last_updated_at.isoformat() if last_updated_at else None,
        }
        update_conversation_metadata(conversation_id, metadata_content)


def _last_updated_at_key(conversation: ConversationMetadata) -> float:
    last_updated_at = conversation.last_updated_at
    if last_updated_at is None:
        return 0.0
    return last_updated_at.timestamp()


def _get_id_from_filename(filename: str) -> int:
    try:
        return int(filename.split('/')[-1].split('.')[0])
    except ValueError:
        logger.warning(f'get id from filename ({filename}) failed.')
        return -1
