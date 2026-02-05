# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import os

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from openhands.core.logger import openhands_logger as logger


class SandboxConfig(BaseModel):
    """Configuration for the sandbox.

    Attributes:
        remote_runtime_api_url: The hostname for the Remote Runtime API.
        local_runtime_url: The default hostname for the local runtime. You may want to change to http://host.docker.internal for DIND environments
        base_container_image: The base container image from which to build the runtime image.
        runtime_container_image: The runtime container image to use.
        user_id: The user ID for the sandbox.
        timeout: The timeout for the default sandbox action execution.
        remote_runtime_init_timeout: The timeout for the remote runtime to start.
        remote_runtime_api_timeout: The timeout for the remote runtime API requests.
        remote_runtime_enable_retries: Whether to enable retries (on recoverable errors like requests.ConnectionError) for the remote runtime API requests.
        enable_auto_lint: Whether to enable auto-lint.
        use_host_network: Whether to use the host network.
        additional_networks: A list of additional Docker networks to connect to
        runtime_binding_address: The binding address for the runtime ports.  It specifies which network interface on the host machine Docker should bind the runtime ports to.
        initialize_plugins: Whether to initialize plugins.
        force_rebuild_runtime: Whether to force rebuild the runtime image.
        runtime_extra_deps: The extra dependencies to install in the runtime image (typically used for evaluation).
            This will be rendered into the end of the Dockerfile that builds the runtime image.
            It can contain any valid shell commands (e.g., pip install numpy).
            The path to the interpreter is available as $OH_INTERPRETER_PATH,
            which can be used to install dependencies for the OH-specific Python interpreter.
        runtime_startup_env_vars: The environment variables to set at the launch of the runtime.
            This is a dictionary of key-value pairs.
            This is useful for setting environment variables that are needed by the runtime.
            For example, for specifying the base url of website for browsergym evaluation.
        browsergym_eval_env: The BrowserGym environment to use for evaluation.
            Default is None for general purpose browsing. Check evaluation/miniwob and evaluation/webarena for examples.
        platform: The platform on which the image should be built. Default is None.
        remote_runtime_resource_factor: Factor to scale the resource allocation for remote runtime.
            Must be one of [1, 2, 4, 8]. Will only be used if the runtime is remote.
        enable_gpu: Whether to enable GPU.
        docker_runtime_kwargs: Additional keyword arguments to pass to the Docker runtime when running containers.
            This should be a Python dictionary literal string that will be parsed into a dictionary.
        trusted_dirs: List of directories that can be trusted to run the OpenHands CLI.
        vscode_port: The port to use for VSCode. If None, a random port will be chosen.
            This is useful when deploying OpenHands in a remote machine where you need to expose a specific port.
    """

    remote_runtime_api_url: str | None = Field(default='http://localhost:8000')
    local_runtime_url: str = Field(default='http://localhost')
    keep_runtime_alive: bool = Field(default=False)
    pause_closed_runtimes: bool = Field(default=True)
    rm_all_containers: bool = Field(default=False)
    api_key: str | None = Field(default=None)
    base_container_image: str | None = Field(
        default='nikolaik/python-nodejs:python3.12-nodejs22'
    )
    runtime_container_image: str | None = Field(default=None)
    user_id: int = Field(default=os.getuid() if hasattr(os, 'getuid') else 1000)
    logger.debug(f'SandboxConfig user_id default: {user_id}')
    timeout: int = Field(default=120)
    remote_runtime_init_timeout: int = Field(default=180)
    remote_runtime_api_timeout: int = Field(default=180)
    remote_runtime_enable_retries: bool = Field(default=True)
    remote_runtime_class: str | None = Field(
        default=None
    )  # can be "None" (default to gvisor) or "sysbox" (support docker inside runtime + more stable)
    enable_auto_lint: bool = Field(
        default=False
    )  # once enabled, OpenHands would lint files after editing
    use_host_network: bool = Field(default=False)
    additional_networks: list[str] = Field(default=[])
    runtime_binding_address: str = Field(default='0.0.0.0')
    runtime_extra_build_args: list[str] | None = Field(default=None)
    initialize_plugins: bool = Field(default=True)
    force_rebuild_runtime: bool = Field(default=False)
    runtime_extra_deps: str | None = Field(default=None)
    runtime_startup_env_vars: dict[str, str] = Field(default_factory=dict)
    browsergym_eval_env: str | None = Field(default=None)
    platform: str | None = Field(default=None)
    close_delay: int = Field(
        default=3600,
        description='The delay in seconds before closing the sandbox after the agent is done.',
    )
    remote_runtime_resource_factor: int = Field(default=1)
    enable_gpu: bool = Field(default=False)
    docker_runtime_kwargs: dict | None = Field(default=None)
    selected_repo: str | None = Field(default=None)
    trusted_dirs: list[str] = Field(default_factory=list)
    vscode_port: int | None = Field(default=None)
    volumes: str | None = Field(
        default=None,
        description="Volume mounts in the format 'host_path:container_path[:mode]', e.g. '/my/host/dir:/workspace:rw'. Multiple mounts can be specified using commas, e.g. '/path1:/workspace/path1,/path2:/workspace/path2:ro'",
    )

    cuda_visible_devices: str | None = Field(default=None)
    model_config = ConfigDict(extra='forbid')

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, 'SandboxConfig']:
        """Create a mapping of SandboxConfig instances from a toml dictionary representing the [sandbox] section.

        The configuration is built from all keys in data.

        Returns:
            dict[str, SandboxConfig]: A mapping where the key "sandbox" corresponds to the [sandbox] configuration
        """
        # Initialize the result mapping
        sandbox_mapping: dict[str, SandboxConfig] = {}

        # Try to create the configuration instance
        try:
            sandbox_mapping['sandbox'] = cls.model_validate(data)
        except ValidationError as e:
            raise ValueError(f'Invalid sandbox configuration: {e}')

        return sandbox_mapping

    @model_validator(mode='after')
    def set_default_base_image(self) -> 'SandboxConfig':
        if self.base_container_image is None:
            self.base_container_image = 'nikolaik/python-nodejs:python3.12-nodejs22'
        return self
