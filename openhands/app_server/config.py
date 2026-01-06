"""Configuration for the OpenHands App Server."""

import os
from pathlib import Path
from typing import AsyncContextManager

import httpx
from fastapi import Depends, Request
from pydantic import Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

# Import the event_callback module to ensure all processors are registered
import openhands.app_server.event_callback  # noqa: F401
from openhands.agent_server.env_parser import from_env
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
    AppConversationInfoServiceInjector,
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
    AppConversationServiceInjector,
)
from openhands.app_server.app_conversation.app_conversation_start_task_service import (
    AppConversationStartTaskService,
    AppConversationStartTaskServiceInjector,
)
from openhands.app_server.app_lifespan.app_lifespan_service import AppLifespanService
from openhands.app_server.app_lifespan.oss_app_lifespan_service import (
    OssAppLifespanService,
)
from openhands.app_server.event.event_service import EventService, EventServiceInjector
from openhands.app_server.event_callback.event_callback_service import (
    EventCallbackService,
    EventCallbackServiceInjector,
)
from openhands.app_server.sandbox.sandbox_service import (
    SandboxService,
    SandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
    SandboxSpecServiceInjector,
)
from openhands.app_server.services.db_session_injector import (
    DbSessionInjector,
)
from openhands.app_server.services.httpx_client_injector import HttpxClientInjector
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.services.jwt_service import JwtService, JwtServiceInjector
from openhands.app_server.user.user_context import UserContext, UserContextInjector
from openhands.sdk.utils.models import OpenHandsModel


def get_default_persistence_dir() -> Path:
    # Recheck env because this function is also used to generate other defaults
    persistence_dir = os.getenv('OH_PERSISTENCE_DIR')

    if persistence_dir:
        result = Path(persistence_dir)
    else:
        result = Path.home() / '.openhands'

    result.mkdir(parents=True, exist_ok=True)
    return result


def get_default_web_url() -> str | None:
    """Get legacy web host parameter.

    If present, we assume we are running under https.
    """
    web_host = os.getenv('WEB_HOST')
    if not web_host:
        return None
    return f'https://{web_host}'


def get_openhands_provider_base_url() -> str | None:
    """Return the base URL for the OpenHands provider, if configured."""
    return os.getenv('OPENHANDS_PROVIDER_BASE_URL') or None


def _get_default_lifespan():
    # Check legacy parameters for saas mode. If we are in SAAS mode do not apply
    # OSS alembic migrations
    if 'saas' in (os.getenv('OPENHANDS_CONFIG_CLS') or '').lower():
        return None
    return OssAppLifespanService()


class AppServerConfig(OpenHandsModel):
    persistence_dir: Path = Field(default_factory=get_default_persistence_dir)
    web_url: str | None = Field(
        default_factory=get_default_web_url,
        description='The URL where OpenHands is running (e.g., http://localhost:3000)',
    )
    openhands_provider_base_url: str | None = Field(
        default_factory=get_openhands_provider_base_url,
        description='Base URL for the OpenHands provider',
    )
    # Dependency Injection Injectors
    event: EventServiceInjector | None = None
    event_callback: EventCallbackServiceInjector | None = None
    sandbox: SandboxServiceInjector | None = None
    sandbox_spec: SandboxSpecServiceInjector | None = None
    app_conversation_info: AppConversationInfoServiceInjector | None = None
    app_conversation_start_task: AppConversationStartTaskServiceInjector | None = None
    app_conversation: AppConversationServiceInjector | None = None
    user: UserContextInjector | None = None
    jwt: JwtServiceInjector | None = None
    httpx: HttpxClientInjector = Field(default_factory=HttpxClientInjector)
    db_session: DbSessionInjector = Field(
        default_factory=lambda: DbSessionInjector(
            persistence_dir=get_default_persistence_dir()
        )
    )

    # Services
    lifespan: AppLifespanService | None = Field(default_factory=_get_default_lifespan)


def config_from_env() -> AppServerConfig:
    # Import defaults...
    from openhands.app_server.app_conversation.live_status_app_conversation_service import (  # noqa: E501
        LiveStatusAppConversationServiceInjector,
    )
    from openhands.app_server.app_conversation.sql_app_conversation_info_service import (  # noqa: E501
        SQLAppConversationInfoServiceInjector,
    )
    from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (  # noqa: E501
        SQLAppConversationStartTaskServiceInjector,
    )
    from openhands.app_server.event.filesystem_event_service import (
        FilesystemEventServiceInjector,
    )
    from openhands.app_server.event_callback.sql_event_callback_service import (
        SQLEventCallbackServiceInjector,
    )
    from openhands.app_server.sandbox.docker_sandbox_service import (
        DockerSandboxServiceInjector,
    )
    from openhands.app_server.sandbox.docker_sandbox_spec_service import (
        DockerSandboxSpecServiceInjector,
    )
    from openhands.app_server.sandbox.process_sandbox_service import (
        ProcessSandboxServiceInjector,
    )
    from openhands.app_server.sandbox.process_sandbox_spec_service import (
        ProcessSandboxSpecServiceInjector,
    )
    from openhands.app_server.sandbox.remote_sandbox_service import (
        RemoteSandboxServiceInjector,
    )
    from openhands.app_server.sandbox.remote_sandbox_spec_service import (
        RemoteSandboxSpecServiceInjector,
    )
    from openhands.app_server.user.auth_user_context import (
        AuthUserContextInjector,
    )

    config: AppServerConfig = from_env(AppServerConfig, 'OH')  # type: ignore

    if config.event is None:
        config.event = FilesystemEventServiceInjector()

    if config.event_callback is None:
        config.event_callback = SQLEventCallbackServiceInjector()

    if config.sandbox is None:
        # Legacy fallback
        if os.getenv('RUNTIME') == 'remote':
            config.sandbox = RemoteSandboxServiceInjector(
                api_key=os.environ['SANDBOX_API_KEY'],
                api_url=os.environ['SANDBOX_REMOTE_RUNTIME_API_URL'],
            )
        elif os.getenv('RUNTIME') in ('local', 'process'):
            config.sandbox = ProcessSandboxServiceInjector()
        else:
            # Support legacy environment variables for Docker sandbox configuration
            docker_sandbox_kwargs: dict = {}
            if os.getenv('SANDBOX_HOST_PORT'):
                docker_sandbox_kwargs['host_port'] = int(
                    os.environ['SANDBOX_HOST_PORT']
                )
            if os.getenv('SANDBOX_CONTAINER_URL_PATTERN'):
                docker_sandbox_kwargs['container_url_pattern'] = os.environ[
                    'SANDBOX_CONTAINER_URL_PATTERN'
                ]
            config.sandbox = DockerSandboxServiceInjector(**docker_sandbox_kwargs)

    if config.sandbox_spec is None:
        if os.getenv('RUNTIME') == 'remote':
            config.sandbox_spec = RemoteSandboxSpecServiceInjector()
        elif os.getenv('RUNTIME') in ('local', 'process'):
            config.sandbox_spec = ProcessSandboxSpecServiceInjector()
        else:
            config.sandbox_spec = DockerSandboxSpecServiceInjector()

    if config.app_conversation_info is None:
        config.app_conversation_info = SQLAppConversationInfoServiceInjector()

    if config.app_conversation_start_task is None:
        config.app_conversation_start_task = (
            SQLAppConversationStartTaskServiceInjector()
        )

    if config.app_conversation is None:
        tavily_api_key = None
        tavily_api_key_str = os.getenv('TAVILY_API_KEY') or os.getenv('SEARCH_API_KEY')
        if tavily_api_key_str:
            tavily_api_key = SecretStr(tavily_api_key_str)
        config.app_conversation = LiveStatusAppConversationServiceInjector(
            tavily_api_key=tavily_api_key
        )

    if config.user is None:
        config.user = AuthUserContextInjector()

    if config.jwt is None:
        config.jwt = JwtServiceInjector(persistence_dir=config.persistence_dir)

    return config


_global_config: AppServerConfig | None = None


def get_global_config() -> AppServerConfig:
    """Get the default local server config shared across the server."""
    global _global_config
    if _global_config is None:
        # Load configuration from environment...
        _global_config = config_from_env()

    return _global_config  # type: ignore


def get_event_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[EventService]:
    injector = get_global_config().event
    assert injector is not None
    return injector.context(state, request)


def get_event_callback_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[EventCallbackService]:
    injector = get_global_config().event_callback
    assert injector is not None
    return injector.context(state, request)


def get_sandbox_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[SandboxService]:
    injector = get_global_config().sandbox
    assert injector is not None
    return injector.context(state, request)


def get_sandbox_spec_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[SandboxSpecService]:
    injector = get_global_config().sandbox_spec
    assert injector is not None
    return injector.context(state, request)


def get_app_conversation_info_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[AppConversationInfoService]:
    injector = get_global_config().app_conversation_info
    assert injector is not None
    return injector.context(state, request)


def get_app_conversation_start_task_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[AppConversationStartTaskService]:
    injector = get_global_config().app_conversation_start_task
    assert injector is not None
    return injector.context(state, request)


def get_app_conversation_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[AppConversationService]:
    injector = get_global_config().app_conversation
    assert injector is not None
    return injector.context(state, request)


def get_user_context(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[UserContext]:
    injector = get_global_config().user
    assert injector is not None
    return injector.context(state, request)


def get_httpx_client(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[httpx.AsyncClient]:
    return get_global_config().httpx.context(state, request)


def get_jwt_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[JwtService]:
    injector = get_global_config().jwt
    assert injector is not None
    return injector.context(state, request)


def get_db_session(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[AsyncSession]:
    return get_global_config().db_session.context(state, request)


def get_app_lifespan_service() -> AppLifespanService | None:
    config = get_global_config()
    return config.lifespan


def depends_event_service():
    injector = get_global_config().event
    assert injector is not None
    return Depends(injector.depends)


def depends_event_callback_service():
    injector = get_global_config().event_callback
    assert injector is not None
    return Depends(injector.depends)


def depends_sandbox_service():
    injector = get_global_config().sandbox
    assert injector is not None
    return Depends(injector.depends)


def depends_sandbox_spec_service():
    injector = get_global_config().sandbox_spec
    assert injector is not None
    return Depends(injector.depends)


def depends_app_conversation_info_service():
    injector = get_global_config().app_conversation_info
    assert injector is not None
    return Depends(injector.depends)


def depends_app_conversation_start_task_service():
    injector = get_global_config().app_conversation_start_task
    assert injector is not None
    return Depends(injector.depends)


def depends_app_conversation_service():
    injector = get_global_config().app_conversation
    assert injector is not None
    return Depends(injector.depends)


def depends_user_context():
    injector = get_global_config().user
    assert injector is not None
    return Depends(injector.depends)


def depends_httpx_client():
    return Depends(get_global_config().httpx.depends)


def depends_jwt_service():
    injector = get_global_config().jwt
    assert injector is not None
    return Depends(injector.depends)


def depends_db_session():
    return Depends(get_global_config().db_session.depends)
