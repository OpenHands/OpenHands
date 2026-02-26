# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import json
import logging
import os
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
import tenacity
from tenacity import RetryCallState

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeError,
    AgentRuntimeNotFoundError,
    AgentRuntimeNotReadyError,
    AgentRuntimeUnavailableError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.builder.remote import RemoteRuntimeBuilder
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.runtime.utils.command import (
    DEFAULT_MAIN_MODULE,
    get_action_execution_server_startup_command,
)
from openhands.runtime.utils.request import send_request
from openhands.runtime.utils.runtime_build import build_runtime_image
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.tenacity_stop import stop_if_should_exit


class RemoteRuntime(ActionExecutionClient):
    """This runtime will connect to a remote oh-runtime-client."""

    port: int = 60000  # default port for the remote runtime client
    runtime_id: str | None = None
    runtime_url: str | None = None
    _runtime_initialized: bool = False
    runtime_builder: RemoteRuntimeBuilder
    container_image: str
    available_hosts: dict[str, int]
    main_module: str

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable[..., None] | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
        main_module: str = DEFAULT_MAIN_MODULE,
    ) -> None:
        super().__init__(
            config,
            event_stream,
            llm_registry,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
            user_id,
            git_provider_tokens,
        )
        logger.debug(f'RemoteRuntime.init user_id {user_id}')
        if self.config.sandbox.api_key is None:
            raise ValueError(
                'API key is required to use the remote runtime. '
                'Please set the API key in the config (config.toml) or as an environment variable (SANDBOX_API_KEY).'
            )
        self.session.headers.update({'X-API-Key': self.config.sandbox.api_key})

        if self.config.workspace_base is not None:
            self.log(
                'debug',
                'Setting workspace_base is not supported in the remote runtime.',
            )
        if self.config.sandbox.remote_runtime_api_url is None:
            raise ValueError(
                'remote_runtime_api_url is required in the remote runtime.'
            )

        assert self.config.sandbox.remote_runtime_class in (None, 'sysbox', 'gvisor')
        self.main_module = main_module

        self.runtime_builder = RemoteRuntimeBuilder(
            self.config.sandbox.remote_runtime_api_url,
            self.config.sandbox.api_key,
            self.session,
        )
        self.available_hosts: dict[str, int] = {}
        self._session_api_key: str | None = None

    def log(self, level: str, message: str, exc_info: bool | None = None) -> None:
        getattr(logger, level)(
            message,
            stacklevel=2,
            exc_info=exc_info,
            extra={
                'session_id': self.sid,
                'runtime_id': self.runtime_id,
            },
        )

    @property
    def action_execution_server_url(self) -> str:
        if self.runtime_url is None:
            raise NotImplementedError('Runtime URL is not initialized')
        return self.runtime_url

    async def connect(self) -> None:
        try:
            await call_sync_from_async(self._start_or_attach_to_runtime)
        except Exception:
            self.close()
            self.log('error', 'Runtime failed to start', exc_info=True)
            raise
        await call_sync_from_async(self.setup_initial_env)
        self._runtime_initialized = True

    def _start_or_attach_to_runtime(self) -> None:
        self.log('info', 'Starting or attaching to runtime')
        existing_runtime = self._check_existing_runtime()
        if existing_runtime:
            self.log('info', f'Using existing runtime with ID: {self.runtime_id}')
        elif self.attach_to_existing:
            self.log('info', f'Failed to find existing runtime for SID: {self.sid}')
            raise AgentRuntimeNotFoundError(
                f'Could not find existing runtime for SID: {self.sid}'
            )
        else:
            self.log('info', 'No existing runtime found, starting a new one')
            if self.config.sandbox.runtime_container_image is None:
                self.log(
                    'info',
                    f'Building remote runtime with base image: {self.config.sandbox.base_container_image}',
                )
                self._build_runtime()
            else:
                self.log(
                    'info',
                    f'Starting remote runtime with image: {self.config.sandbox.runtime_container_image}',
                )
                self.container_image = self.config.sandbox.runtime_container_image
            self._start_runtime()
        assert self.runtime_id is not None, (
            'Runtime ID is not set. This should never happen.'
        )
        assert self.runtime_url is not None, (
            'Runtime URL is not set. This should never happen.'
        )
        if not self.attach_to_existing:
            self.log('info', 'Waiting for runtime to be alive...')
        self._wait_until_alive()
        if not self.attach_to_existing:
            self.log('info', 'Runtime is ready.')
        self.set_runtime_status(RuntimeStatus.READY)

    def _check_existing_runtime(self) -> bool:
        self.log('info', f'Checking for existing runtime with session ID: {self.sid}')
        try:
            response = self._send_runtime_api_request(
                'GET',
                f'{self.config.sandbox.remote_runtime_api_url}/sessions/{self.sid}',
            )
            data = response.json()
            status = data.get('status')
            self.log('info', f'Found runtime with status: {status}')
            if status == 'running' or status == 'paused':
                self._parse_runtime_response(response)
        except httpx.HTTPError as e:
            if e.response.status_code == 404:
                self.log(
                    'info', f'No existing runtime found for session ID: {self.sid}'
                )
                return False
            self.log('error', f'Error while looking for remote runtime: {e}')
            raise
        except json.decoder.JSONDecodeError as e:
            self.log(
                'error',
                f'Invalid JSON response from runtime API: {e}. URL: {self.config.sandbox.remote_runtime_api_url}/sessions/{self.sid}. Response: {response}',
            )
            raise

        if status == 'running':
            self.log('info', 'Found existing runtime in running state')
            return True
        elif status == 'stopped':
            self.log('info', 'Found existing runtime, but it is stopped')
            return False
        elif status == 'paused':
            self.log(
                'info', 'Found existing runtime in paused state, attempting to resume'
            )
            try:
                self._resume_runtime()
                self.log('info', 'Successfully resumed paused runtime')
                return True
            except Exception as e:
                self.log(
                    'error', f'Failed to resume paused runtime: {e}', exc_info=True
                )
                # Return false to indicate we couldn't use the existing runtime
                return False
        else:
            self.log('error', f'Invalid response from runtime API: {data}')
            return False

    def _build_runtime(self) -> None:
        self.log('debug', f'Building RemoteRuntime config:\n{self.config}')
        self.set_runtime_status(RuntimeStatus.BUILDING_RUNTIME)
        response = self._send_runtime_api_request(
            'GET',
            f'{self.config.sandbox.remote_runtime_api_url}/registry_prefix',
        )
        response_json = response.json()
        registry_prefix = response_json['registry_prefix']
        os.environ['OH_RUNTIME_RUNTIME_IMAGE_REPO'] = (
            registry_prefix.rstrip('/') + '/runtime'
        )
        self.log(
            'debug',
            f'Runtime image repo: {os.environ["OH_RUNTIME_RUNTIME_IMAGE_REPO"]}',
        )
        if self.config.sandbox.base_container_image is None:
            raise ValueError(
                'base_container_image is required to build the runtime image. '
            )
        if self.config.sandbox.runtime_extra_deps:
            self.log(
                'debug',
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}',
            )

        # Build the container image
        self.container_image = build_runtime_image(
            self.config.sandbox.base_container_image,
            self.runtime_builder,
            platform=self.config.sandbox.platform,
            extra_deps=self.config.sandbox.runtime_extra_deps,
            force_rebuild=self.config.sandbox.force_rebuild_runtime,
            enable_browser=self.config.enable_browser,
        )

        response = self._send_runtime_api_request(
            'GET',
            f'{self.config.sandbox.remote_runtime_api_url}/image_exists',
            params={'image': self.container_image},
        )
        if not response.json()['exists']:
            raise AgentRuntimeError(
                f'Container image {self.container_image} does not exist'
            )

    def _start_runtime(self) -> None:
        # Prepare the request body for the /start endpoint
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
        command = self.get_action_execution_server_startup_command()
        environment: dict[str, str] = {}
        if self.config.debug or os.environ.get('DEBUG', 'false').lower() == 'true':
            environment['DEBUG'] = 'true'
        environment.update(self.config.sandbox.runtime_startup_env_vars)
        start_request: dict[str, Any] = {
            'image': self.container_image,
            'command': command,
            'working_dir': '/openhands/code/',
            'environment': environment,
            'session_id': self.sid,
            'resource_factor': self.config.sandbox.remote_runtime_resource_factor,
        }
        if self.config.sandbox.remote_runtime_class == 'sysbox':
            start_request['runtime_class'] = 'sysbox-runc'
        # We ignore other runtime classes for now, because both None and 'gvisor' map to 'gvisor'

        # Start the sandbox using the /start endpoint
        try:
            response = self._send_runtime_api_request(
                'POST',
                f'{self.config.sandbox.remote_runtime_api_url}/start',
                json=start_request,
            )
            self._parse_runtime_response(response)
            self.log(
                'debug',
                f'Runtime started. URL: {self.runtime_url}',
            )
        except httpx.HTTPError as e:
            self.log('error', f'Unable to start runtime: {str(e)}')
            raise AgentRuntimeUnavailableError() from e

    def _resume_runtime(self) -> None:
        """Resume a stopped runtime.

        Steps:
        1. Show status update that runtime is being started.
        2. Send the runtime API a /resume request
        3. Poll for the runtime to be ready
        4. Update env vars
        """
        self.log('info', f'Attempting to resume runtime with ID: {self.runtime_id}')
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
        try:
            response = self._send_runtime_api_request(
                'POST',
                f'{self.config.sandbox.remote_runtime_api_url}/resume',
                json={'runtime_id': self.runtime_id},
            )
            self.log(
                'info',
                f'Resume API call successful with status code: {response.status_code}',
            )
        except Exception as e:
            self.log('error', f'Failed to call /resume API: {e}', exc_info=True)
            raise

        self.log(
            'info', 'Runtime resume API call completed, waiting for it to be alive...'
        )
        try:
            self._wait_until_alive()
            self.log('info', 'Runtime is now alive after resume')
        except Exception as e:
            self.log(
                'error',
                f'Runtime failed to become alive after resume: {e}',
                exc_info=True,
            )
            raise

        try:
            self.setup_initial_env()
            self.log('info', 'Successfully set up initial environment after resume')
        except Exception as e:
            self.log(
                'error',
                f'Failed to set up initial environment after resume: {e}',
                exc_info=True,
            )
            raise

        self.log('info', 'Runtime successfully resumed and alive.')

    def _parse_runtime_response(self, response: httpx.Response) -> None:
        start_response = response.json()
        self.runtime_id = start_response['runtime_id']
        self.runtime_url = start_response['url']
        self.available_hosts = start_response.get('work_hosts', {})

        if 'session_api_key' in start_response:
            self.session.headers.update(
                {'X-Session-API-Key': start_response['session_api_key']}
            )
            self._session_api_key = start_response['session_api_key']
            self.log(
                'debug',
                'Session API key set',
            )

    @property
    def session_api_key(self) -> str | None:
        return self._session_api_key

    @property
    def vscode_url(self) -> str | None:
        token = super().get_vscode_token()
        if not token:
            return None
        assert self.runtime_url is not None and self.runtime_id is not None
        self.log('debug', f'runtime_url: {self.runtime_url}')
        parsed = urlparse(self.runtime_url)
        scheme, netloc, path = parsed.scheme, parsed.netloc, parsed.path or '/'
        # Path mode if runtime_url path starts with /{id}
        path_mode = path.startswith(f'/{self.runtime_id}')
        if path_mode:
            vscode_url = f'{scheme}://{netloc}/{self.runtime_id}/vscode?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        else:
            vscode_url = f'{scheme}://vscode-{netloc}/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        self.log(
            'debug',
            f'VSCode URL: {vscode_url}',
        )
        return vscode_url

    @property
    def web_hosts(self) -> dict[str, int]:
        return self.available_hosts

    def _wait_until_alive(self) -> None:
        retry_decorator = tenacity.retry(
            stop=tenacity.stop_after_delay(
                self.config.sandbox.remote_runtime_init_timeout
            )
            | stop_if_should_exit()
            | self._stop_if_closed,
            reraise=True,
            retry=tenacity.retry_if_exception_type(AgentRuntimeNotReadyError),
            wait=tenacity.wait_fixed(2),
        )
        retry_decorator(self._wait_until_alive_impl)()

    def _wait_until_alive_impl(self) -> None:
        self.log('debug', f'Waiting for runtime to be alive at url: {self.runtime_url}')
        self.log(
            'debug',
            f'Sending request to: {self.config.sandbox.remote_runtime_api_url}/runtime/{self.runtime_id}',
        )
        runtime_info_response = self._send_runtime_api_request(
            'GET',
            f'{self.config.sandbox.remote_runtime_api_url}/runtime/{self.runtime_id}',
        )
        runtime_data = runtime_info_response.json()
        self.log(
            'debug',
            f'received response: {runtime_data}',
        )
        assert 'runtime_id' in runtime_data
        assert runtime_data['runtime_id'] == self.runtime_id
        assert 'pod_status' in runtime_data
        pod_status = runtime_data['pod_status'].lower()
        self.log('debug', f'Pod status: {pod_status}')
        restart_count = runtime_data.get('restart_count', 0)
        if restart_count != 0:
            restart_reasons = runtime_data.get('restart_reasons')
            self.log(
                'debug', f'Pod restarts: {restart_count}, reasons: {restart_reasons}'
            )

        # FIXME: We should fix it at the backend of /start endpoint, make sure
        # the pod is created before returning the response.
        # Retry a period of time to give the cluster time to start the pod
        if pod_status == 'ready':
            try:
                self.check_if_alive()
            except httpx.HTTPError as e:
                self.log(
                    'warning',
                    f"Runtime /alive failed, but pod says it's ready: {str(e)}",
                )
                raise AgentRuntimeNotReadyError(
                    f'Runtime /alive failed to respond with 200: {str(e)}'
                )
            return
        elif (
            pod_status == 'not found'
            or pod_status == 'pending'
            or pod_status == 'running'
        ):  # nb: Running is not yet Ready
            raise AgentRuntimeNotReadyError(
                f'Runtime (ID={self.runtime_id}) is not yet ready. Status: {pod_status}'
            )
        elif pod_status in ('failed', 'unknown', 'crashloopbackoff'):
            if pod_status == 'crashloopbackoff':
                raise AgentRuntimeUnavailableError(
                    'Runtime crashed and is being restarted, potentially due to memory usage. Please try again.'
                )
            else:
                raise AgentRuntimeUnavailableError(
                    f'Runtime is unavailable (status: {pod_status}). Please try again.'
                )
        else:
            # Maybe this should be a hard failure, but passing through in case the API changes
            self.log('warning', f'Unknown pod status: {pod_status}')

        self.log(
            'debug',
            f'Waiting for runtime pod to be active. Current status: {pod_status}',
        )
        raise AgentRuntimeNotReadyError()

    def close(self) -> None:
        if self.attach_to_existing:
            super().close()
            return
        if self.config.sandbox.keep_runtime_alive:
            if self.config.sandbox.pause_closed_runtimes:
                try:
                    if not self._runtime_closed:
                        self._send_runtime_api_request(
                            'POST',
                            f'{self.config.sandbox.remote_runtime_api_url}/pause',
                            json={'runtime_id': self.runtime_id},
                        )
                        self.log('info', 'Runtime paused.')
                except Exception as e:
                    self.log('error', f'Unable to pause runtime: {str(e)}')
                    raise e
            super().close()
            return
        try:
            if not self._runtime_closed:
                self._send_runtime_api_request(
                    'POST',
                    f'{self.config.sandbox.remote_runtime_api_url}/stop',
                    json={'runtime_id': self.runtime_id},
                )
                self.log('info', 'Runtime stopped.')
        except Exception as e:
            self.log('error', f'Unable to stop runtime: {str(e)}')
            raise e
        finally:
            super().close()

    def _send_runtime_api_request(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        try:
            kwargs['timeout'] = self.config.sandbox.remote_runtime_api_timeout
            return send_request(self.session, method, url, **kwargs)
        except httpx.TimeoutException:
            self.log(
                'error',
                f'No response received within the timeout period for url: {url}',
            )
            raise

    def _send_action_server_request(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        if not self.config.sandbox.remote_runtime_enable_retries:
            return self._send_action_server_request_impl(method, url, **kwargs)

        retry_decorator = tenacity.retry(
            retry=tenacity.retry_if_exception_type(httpx.NetworkError),
            stop=tenacity.stop_after_attempt(3)
            | stop_if_should_exit()
            | self._stop_if_closed,
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        )
        return retry_decorator(self._send_action_server_request_impl)(
            method, url, **kwargs
        )

    def _send_action_server_request_impl(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        try:
            return super()._send_action_server_request(method, url, **kwargs)
        except httpx.TimeoutException:
            self.log(
                'error',
                f'No response received within the timeout period for url: {url}',
            )
            raise

        except httpx.HTTPError as e:
            if hasattr(e, 'response') and e.response.status_code in (404, 502, 504):
                if e.response.status_code == 404:
                    raise AgentRuntimeDisconnectedError(
                        f'Runtime is not responding. This may be temporary, please try again. Original error: {e}'
                    ) from e
                else:  # 502, 504
                    raise AgentRuntimeDisconnectedError(
                        f'Runtime is temporarily unavailable. This may be due to a restart or network issue, please try again. Original error: {e}'
                    ) from e
            elif hasattr(e, 'response') and e.response.status_code == 503:
                if self.config.sandbox.keep_runtime_alive:
                    self.log(
                        'info',
                        f'Runtime appears to be paused (503 response). Runtime ID: {self.runtime_id}, URL: {url}',
                    )
                    try:
                        self._resume_runtime()
                        self.log(
                            'info', 'Successfully resumed runtime after 503 response'
                        )
                        return super()._send_action_server_request(
                            method, url, **kwargs
                        )
                    except Exception as resume_error:
                        self.log(
                            'error',
                            f'Failed to resume runtime after 503 response: {resume_error}',
                            exc_info=True,
                        )
                        raise AgentRuntimeDisconnectedError(
                            f'Runtime is paused and could not be resumed. Original error: {e}, Resume error: {resume_error}'
                        ) from resume_error
                else:
                    self.log(
                        'info',
                        'Runtime appears to be paused (503 response) but keep_runtime_alive is False',
                    )
                    raise AgentRuntimeDisconnectedError(
                        f'Runtime is temporarily unavailable. This may be due to a restart or network issue, please try again. Original error: {e}'
                    ) from e
            else:
                raise e

    def _stop_if_closed(self, retry_state: RetryCallState) -> bool:
        return self._runtime_closed

    def get_action_execution_server_startup_command(self):
        return get_action_execution_server_startup_command(
            server_port=self.port,
            plugins=self.plugins,
            app_config=self.config,
            main_module=self.main_module,
        )
