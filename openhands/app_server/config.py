"""Configuration for the OpenHands App Server."""

import logging
import os
from pathlib import Path
from typing import AsyncContextManager

import httpx
import tomllib
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
from openhands.app_server.pending_messages.pending_message_service import (
    PendingMessageService,
    PendingMessageServiceInjector,
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
from openhands.app_server.web_client.default_web_client_config_injector import (
    DefaultWebClientConfigInjector,
)
from openhands.app_server.web_client.web_client_config_injector import (
    WebClientConfigInjector,
)
from openhands.sdk.utils.models import OpenHandsModel
from openhands.server.types import AppMode
from openhands.utils.environment import StorageProvider, get_storage_provider

_logger = logging.getLogger(__name__)


def _env_var_is_set(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value != ''


def _env_has_prefix(prefix: str) -> bool:
    return any(key.startswith(prefix) and _env_var_is_set(key) for key in os.environ)


def _resolve_app_server_toml_path() -> Path | None:
    """Resolve app-server TOML config path.

    Resolution order:
    1. `OH_CONFIG_FILE` or `OPENHANDS_CONFIG_FILE` when set.
    2. `docker.toml` in current working directory.
    3. `config.toml` in current working directory.
    4. `/app/docker.toml`.
    5. `/app/config.toml`.
    """
    explicit_path = os.getenv('OH_CONFIG_FILE') or os.getenv('OPENHANDS_CONFIG_FILE')
    if explicit_path:
        resolved = Path(explicit_path).expanduser()
        if resolved.is_file():
            return resolved
        _logger.warning('Configured app server TOML file was not found: %s', resolved)
        return None

    candidates = [
        Path.cwd() / 'docker.toml',
        Path.cwd() / 'config.toml',
        Path('/app/docker.toml'),
        Path('/app/config.toml'),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _load_app_server_toml() -> dict | None:
    path = _resolve_app_server_toml_path()
    if path is None:
        return None
    try:
        with path.open('rb') as file_handle:
            config = tomllib.load(file_handle)
        _logger.info('Loaded app-server TOML configuration from %s', path)
        return config
    except tomllib.TOMLDecodeError as e:
        _logger.warning('Failed to parse app-server TOML file %s: %s', path, e)
    except OSError as e:
        _logger.warning('Failed to read app-server TOML file %s: %s', path, e)
    return None


def _parse_mounts(mounts_spec: str):
    """Parse legacy volume mount string into docker sandbox mounts."""
    from openhands.app_server.sandbox.docker_sandbox_service import VolumeMount

    mounts = []
    for mount_spec in mounts_spec.split(','):
        mount_spec = mount_spec.strip()
        if not mount_spec:
            continue
        parts = mount_spec.split(':')
        if len(parts) < 2:
            _logger.warning('Skipping invalid mount spec %r in TOML config', mount_spec)
            continue

        host_path = parts[0].strip()
        container_path = parts[1].strip()
        mode = parts[2].strip() if len(parts) > 2 and parts[2].strip() else 'rw'
        mounts.append(
            VolumeMount(
                host_path=host_path,
                container_path=container_path,
                mode=mode,
            )
        )
    return mounts


def _extract_container_ports_from_publish_args(args: list[str]) -> list[int]:
    """Extract container ports from docker publish args (e.g. `-p 8080:8080`)."""
    ports: list[int] = []
    i = 0
    while i < len(args):
        arg = args[i].strip()
        publish_spec: str | None = None

        if arg in ('-p', '--publish') and i + 1 < len(args):
            publish_spec = args[i + 1].strip()
            i += 1
        elif arg.startswith('-p '):
            publish_spec = arg[3:].strip()
        elif arg.startswith('--publish '):
            publish_spec = arg[len('--publish ') :].strip()
        elif arg.startswith('--publish='):
            publish_spec = arg[len('--publish=') :].strip()

        if publish_spec:
            # Support publish specs like:
            # - 8080
            # - 8080:8080
            # - 127.0.0.1:8080:8080/tcp
            container_part = publish_spec.split(':')[-1].split('/')[0].strip()
            if container_part.isdigit():
                port = int(container_part)
                if port not in ports:
                    ports.append(port)

        i += 1

    return ports


def get_default_persistence_dir() -> Path:
    # Recheck env because this function is also used to generate other defaults
    persistence_dir = os.getenv('OH_PERSISTENCE_DIR')

    # Legacy V0 fallback variable
    if persistence_dir is None:
        persistence_dir = os.getenv('FILE_STORE_PATH')

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


def get_default_permitted_cors_origins() -> list[str]:
    """Get permitted CORS origins, falling back to legacy PERMITTED_CORS_ORIGINS env var.

    The preferred configuration is via OH_PERMITTED_CORS_ORIGINS_0, _1, etc.
    (handled by the pydantic from_env parser). This fallback supports the legacy
    comma-separated PERMITTED_CORS_ORIGINS environment variable.
    """
    legacy = os.getenv('PERMITTED_CORS_ORIGINS', '')
    if legacy:
        return [o.strip() for o in legacy.split(',') if o.strip()]
    return []


def get_openhands_provider_base_url() -> str | None:
    """Return the base URL for the OpenHands provider, if configured.

    Falls back to LLM_BASE_URL for backward compatibility.
    """
    return os.getenv('OPENHANDS_PROVIDER_BASE_URL') or os.getenv('LLM_BASE_URL') or None


def _get_default_lifespan():
    # Check legacy parameters for saas mode. If we are in SAAS mode do not apply
    # OpenHands alembic migrations
    if 'saas' in (os.getenv('OPENHANDS_CONFIG_CLS') or '').lower():
        return None
    return OssAppLifespanService()


class AppServerConfig(OpenHandsModel):
    persistence_dir: Path = Field(default_factory=get_default_persistence_dir)
    web_url: str | None = Field(
        default_factory=get_default_web_url,
        description='The URL where OpenHands is running (e.g., http://localhost:3000)',
    )
    permitted_cors_origins: list[str] = Field(
        default_factory=get_default_permitted_cors_origins,
        description=(
            'Additional permitted CORS origins for both the app server and agent '
            'server containers. Configure via OH_PERMITTED_CORS_ORIGINS_0, _1, etc. '
            'Falls back to legacy PERMITTED_CORS_ORIGINS env var.'
        ),
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
    pending_message: PendingMessageServiceInjector | None = None
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
    app_mode: AppMode = AppMode.OPENHANDS
    web_client: WebClientConfigInjector = Field(
        default_factory=DefaultWebClientConfigInjector
    )


def _apply_toml_overrides(config: AppServerConfig) -> None:
    """Apply TOML configuration overrides to app server config.

    TOML values are treated as defaults and are only applied when no environment
    variable override is present.
    """
    toml_config = _load_app_server_toml()
    if not toml_config or not isinstance(toml_config, dict):
        return

    app_server_section = toml_config.get('app_server')
    if isinstance(app_server_section, dict):
        if (
            not _env_var_is_set('OH_WEB_URL')
            and not _env_var_is_set('WEB_HOST')
            and isinstance(app_server_section.get('web_url'), str)
        ):
            config.web_url = app_server_section['web_url']

        if (
            not _env_has_prefix('OH_PERMITTED_CORS_ORIGINS_')
            and not _env_var_is_set('PERMITTED_CORS_ORIGINS')
            and isinstance(app_server_section.get('permitted_cors_origins'), list)
        ):
            configured_origins = [
                item
                for item in app_server_section['permitted_cors_origins']
                if isinstance(item, str)
            ]
            if configured_origins:
                config.permitted_cors_origins = configured_origins

    # Support both the V1 section ([app_server.sandbox]) and legacy section ([sandbox]).
    sandbox_section = None
    if isinstance(app_server_section, dict) and isinstance(
        app_server_section.get('sandbox'), dict
    ):
        sandbox_section = app_server_section['sandbox']
    elif isinstance(toml_config.get('sandbox'), dict):
        sandbox_section = toml_config['sandbox']

    if sandbox_section is None:
        return

    from openhands.app_server.sandbox.docker_sandbox_service import (
        DockerSandboxServiceInjector,
        ExposedPort,
    )

    if not isinstance(config.sandbox, DockerSandboxServiceInjector):
        return

    if (
        not _env_var_is_set('AGENT_SERVER_USE_HOST_NETWORK')
        and not _env_var_is_set('OH_SANDBOX_USE_HOST_NETWORK')
        and isinstance(sandbox_section.get('use_host_network'), bool)
    ):
        config.sandbox.use_host_network = sandbox_section['use_host_network']

    if (
        not _env_var_is_set('SANDBOX_HOST_PORT')
        and not _env_var_is_set('OH_SANDBOX_HOST_PORT')
        and isinstance(sandbox_section.get('host_port'), int)
    ):
        config.sandbox.host_port = sandbox_section['host_port']

    if (
        not _env_var_is_set('SANDBOX_CONTAINER_URL_PATTERN')
        and not _env_var_is_set('OH_SANDBOX_CONTAINER_URL_PATTERN')
        and isinstance(sandbox_section.get('container_url_pattern'), str)
    ):
        config.sandbox.container_url_pattern = sandbox_section['container_url_pattern']

    if (
        not _env_var_is_set('SANDBOX_STARTUP_GRACE_SECONDS')
        and not _env_var_is_set('OH_SANDBOX_STARTUP_GRACE_SECONDS')
        and isinstance(sandbox_section.get('startup_grace_seconds'), int)
    ):
        config.sandbox.startup_grace_seconds = sandbox_section['startup_grace_seconds']

    if not _env_var_is_set('SANDBOX_VOLUMES') and not _env_has_prefix(
        'OH_SANDBOX_MOUNTS_'
    ):
        volumes = sandbox_section.get('volumes')
        if isinstance(volumes, str):
            parsed_mounts = _parse_mounts(volumes)
            if parsed_mounts:
                config.sandbox.mounts = parsed_mounts

    # Compatibility path: support legacy runtime_extra_build_args publish flags
    # by turning them into additional exposed container ports in V1 docker sandboxes.
    if not _env_has_prefix('OH_SANDBOX_EXPOSED_PORTS_') and isinstance(
        sandbox_section.get('runtime_extra_build_args'), list
    ):
        parsed_args = [
            arg
            for arg in sandbox_section['runtime_extra_build_args']
            if isinstance(arg, str)
        ]
        custom_ports = _extract_container_ports_from_publish_args(parsed_args)
        if custom_ports:
            existing_ports = {
                exposed_port.container_port
                for exposed_port in config.sandbox.exposed_ports
            }
            extra_ports = [
                ExposedPort(
                    name=f'CUSTOM_{port}',
                    description='Custom sandbox port from TOML runtime_extra_build_args',
                    container_port=port,
                )
                for port in custom_ports
                if port not in existing_ports
            ]
            if extra_ports:
                config.sandbox.exposed_ports = [
                    *config.sandbox.exposed_ports,
                    *extra_ports,
                ]


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
    from openhands.app_server.event.aws_event_service import (
        AwsEventServiceInjector,
    )
    from openhands.app_server.event.filesystem_event_service import (
        FilesystemEventServiceInjector,
    )
    from openhands.app_server.event.google_cloud_event_service import (
        GoogleCloudEventServiceInjector,
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
        provider = get_storage_provider()

        if provider == StorageProvider.AWS:
            # AWS S3 storage configuration
            bucket_name = os.environ.get('FILE_STORE_PATH')
            if not bucket_name:
                raise ValueError(
                    'FILE_STORE_PATH environment variable is required for S3 storage'
                )
            config.event = AwsEventServiceInjector(bucket_name=bucket_name)
        elif provider == StorageProvider.GCP:
            # Google Cloud storage configuration
            bucket_name = os.environ.get('FILE_STORE_PATH')
            if not bucket_name:
                raise ValueError(
                    'FILE_STORE_PATH environment variable is required for Google Cloud storage'
                )
            config.event = GoogleCloudEventServiceInjector(bucket_name=bucket_name)
        else:
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
            # Allow configuring sandbox startup grace period
            # This is useful for slower machines or cloud environments where
            # the agent-server container takes longer to initialize
            if os.getenv('SANDBOX_STARTUP_GRACE_SECONDS'):
                docker_sandbox_kwargs['startup_grace_seconds'] = int(
                    os.environ['SANDBOX_STARTUP_GRACE_SECONDS']
                )
            # Parse SANDBOX_VOLUMES and convert to VolumeMount objects
            # This is set by the CLI's --mount-cwd flag
            sandbox_volumes = os.getenv('SANDBOX_VOLUMES')
            if sandbox_volumes:
                mounts = _parse_mounts(sandbox_volumes)
                if mounts:
                    docker_sandbox_kwargs['mounts'] = mounts
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

    if config.pending_message is None:
        from openhands.app_server.pending_messages.pending_message_service import (
            SQLPendingMessageServiceInjector,
        )

        config.pending_message = SQLPendingMessageServiceInjector()

    if config.user is None:
        config.user = AuthUserContextInjector()

    if config.jwt is None:
        config.jwt = JwtServiceInjector(persistence_dir=config.persistence_dir)

    _apply_toml_overrides(config)

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


def get_pending_message_service(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[PendingMessageService]:
    injector = get_global_config().pending_message
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


def depends_pending_message_service():
    injector = get_global_config().pending_message
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
