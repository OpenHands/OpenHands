from __future__ import annotations

import asyncio
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from pydantic import SecretStr

from openhands.app_server.app_lifespan.app_lifespan_service import AppLifespanService
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType
from openhands.storage.data_models.secrets import Secrets


class OssAppLifespanService(AppLifespanService):
    run_alembic_on_startup: bool = True

    _idle_timeout_task: asyncio.Task | None = None

    async def __aenter__(self):
        if self.run_alembic_on_startup:
            self.run_alembic()
        await self._seed_provider_tokens_from_env()
        self._pre_pull_sandbox_images()
        self._start_idle_timeout_monitor()
        self._start_tcp_port_forwarder_manager()
        self._start_registry_cache()
        return self

    async def _seed_provider_tokens_from_env(self) -> None:
        """Seed provider tokens from environment variables into the secrets store."""
        github_token = os.environ.get('GITHUB_TOKEN', '')
        if not github_token:
            return

        from openhands.server.user_auth.default_user_auth import DefaultUserAuth

        try:
            user_auth = DefaultUserAuth()
            secrets_store = await user_auth.get_secrets_store()
            existing = await secrets_store.load()

            # Don't overwrite a token already stored by the user
            if existing and existing.provider_tokens:
                gh = existing.provider_tokens.get(ProviderType.GITHUB)
                if gh and gh.token and gh.token.get_secret_value():
                    return

            # Build new secrets, preserving any existing custom secrets
            provider_tokens = {
                ProviderType.GITHUB: ProviderToken(
                    token=SecretStr(github_token),
                ),
            }
            new_secrets = Secrets(
                provider_tokens=provider_tokens,
                custom_secrets=existing.custom_secrets if existing else {},
            )
            await secrets_store.store(new_secrets)
            logger.info('Seeded GitHub token from GITHUB_TOKEN env var')
        except Exception:
            logger.warning(
                'Failed to seed GitHub token from environment', exc_info=True
            )

    def _pre_pull_sandbox_images(self) -> None:
        """Kick off a background pull of sandbox Docker images.

        This is fire-and-forget: the app starts serving immediately while images
        download in the background.  If the pull hasn't finished by the time a
        user starts a conversation, ``inject()`` will await the running task
        instead of starting a duplicate pull.
        """
        from openhands.app_server.config import get_global_config
        from openhands.app_server.sandbox.docker_sandbox_spec_service import (
            DockerSandboxSpecServiceInjector,
        )

        try:
            config = get_global_config()
            injector = config.sandbox_spec
            if isinstance(injector, DockerSandboxSpecServiceInjector):
                injector.start_background_pull()
            else:
                logger.debug(
                    'Sandbox spec injector is not Docker-based; '
                    'skipping background image pull'
                )
        except Exception:
            logger.warning(
                'Failed to start background sandbox image pull', exc_info=True
            )

    def _start_idle_timeout_monitor(self) -> None:
        """Initialise the idle-timeout manager and start the background monitor.

        Reads timeout configuration from the sandbox injector (which pulls
        from environment variables via ``OH_SANDBOX__IDLE_TIMEOUT_SECONDS``
        and ``OH_SANDBOX__IDLE_WARNING_SECONDS``).
        """
        from openhands.app_server.config import get_global_config
        from openhands.app_server.idle_timeout_manager import (
            init_idle_timeout_manager,
        )
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )

        try:
            config = get_global_config()
            injector = config.sandbox

            timeout_seconds = 1800  # default 30 min
            warning_seconds = 300  # default 5 min

            if isinstance(injector, DockerSandboxServiceInjector):
                timeout_seconds = injector.idle_timeout_seconds
                warning_seconds = injector.idle_warning_seconds

            if timeout_seconds <= 0:
                logger.info('Idle timeout disabled (timeout_seconds <= 0)')
                return

            init_idle_timeout_manager(
                timeout_seconds=timeout_seconds,
                warning_seconds=warning_seconds,
            )
            self._idle_timeout_task = asyncio.create_task(self._idle_timeout_loop())
        except Exception:
            logger.warning('Failed to start idle timeout monitor', exc_info=True)

    def _start_tcp_port_forwarder_manager(self) -> None:
        """Initialise the TCP port forwarder manager for Codespace environments.

        Only activates when running inside a GitHub Codespace (``CODESPACE_NAME``
        is set) with a shared Docker network configured on the sandbox injector.
        """
        from openhands.app_server.config import get_global_config
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )
        from openhands.app_server.sandbox.tcp_port_forwarder import (
            TcpPortForwarderManager,
            set_tcp_port_forwarder_manager,
        )

        if not os.getenv('CODESPACE_NAME'):
            return

        try:
            config = get_global_config()
            injector = config.sandbox

            if isinstance(injector, DockerSandboxServiceInjector) and injector.network:
                set_tcp_port_forwarder_manager(TcpPortForwarderManager())
        except Exception:
            logger.warning('Failed to start TCP port forwarder manager', exc_info=True)

    def _start_registry_cache(self) -> None:
        """Start a pull-through registry cache for Docker-in-Docker sandboxes.

        Only activates when the sandbox injector has ``dind_registry_cache=True``
        and ``privileged=True``.  The mirror URL is stored on the injector so
        it gets passed through to each ``DockerSandboxService`` instance.
        """
        from openhands.app_server.config import get_global_config
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )

        try:
            config = get_global_config()
            injector = config.sandbox

            if not isinstance(injector, DockerSandboxServiceInjector):
                return
            if not injector.dind_registry_cache or not injector.privileged:
                return

            from openhands.app_server.sandbox.registry_cache import (
                RegistryCacheManager,
            )

            manager = RegistryCacheManager(port=injector.dind_registry_port)
            mirror_url = manager.ensure_running()
            # Store on the injector so inject() can pass it to DockerSandboxService
            injector._registry_mirror_url = mirror_url  # type: ignore[attr-defined]
            logger.info(f'Registry cache mirror available at {mirror_url}')
        except Exception:
            logger.warning(
                'Failed to start registry cache — DinD will pull directly',
                exc_info=True,
            )

    async def _idle_timeout_loop(self) -> None:
        """Periodically check for idle sandboxes and pause them."""
        from openhands.app_server.config import get_sandbox_service
        from openhands.app_server.idle_timeout_manager import (
            get_idle_timeout_manager,
        )
        from openhands.app_server.services.injector import InjectorState
        from openhands.app_server.user.auth_user_context import AuthUserContext
        from openhands.app_server.user.specifiy_user_context import (
            USER_CONTEXT_ATTR,
        )
        from openhands.server.user_auth.user_auth import get_for_user

        while True:
            try:
                await asyncio.sleep(30)  # check every 30 seconds

                manager = get_idle_timeout_manager()
                if manager is None:
                    continue

                # Log warnings (informational only — the frontend polls for these)
                warned = manager.get_sandboxes_to_warn()
                for sandbox_id in warned:
                    logger.info(
                        f'Idle warning: sandbox {sandbox_id} will be paused in '
                        f'{manager.warning_seconds}s'
                    )

                # Pause timed-out sandboxes
                to_pause = manager.get_sandboxes_to_pause()
                if not to_pause:
                    continue

                # Create a DI context to access the sandbox service
                state = InjectorState()
                user_auth = await get_for_user('root')
                setattr(
                    state,
                    USER_CONTEXT_ATTR,
                    AuthUserContext(user_auth=user_auth),
                )
                async with get_sandbox_service(state) as sandbox_service:
                    for sandbox_id in to_pause:
                        try:
                            logger.info(f'Pausing idle sandbox: {sandbox_id}')
                            await sandbox_service.pause_sandbox(sandbox_id)
                            # remove() is called inside pause_sandbox already
                        except Exception:
                            logger.warning(
                                f'Failed to pause idle sandbox {sandbox_id}',
                                exc_info=True,
                            )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning('Error in idle timeout loop', exc_info=True)

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._idle_timeout_task is not None:
            self._idle_timeout_task.cancel()
            try:
                await self._idle_timeout_task
            except asyncio.CancelledError:
                pass

    def run_alembic(self):
        # Run alembic upgrade head to ensure database is up to date
        alembic_dir = Path(__file__).parent / 'alembic'
        alembic_ini = alembic_dir / 'alembic.ini'

        # Create alembic config with absolute paths
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option('script_location', str(alembic_dir))

        # Change to alembic directory for the command execution
        original_cwd = os.getcwd()
        try:
            os.chdir(str(alembic_dir.parent))
            command.upgrade(alembic_cfg, 'head')
        finally:
            os.chdir(original_cwd)
