# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from openhands.runtime.base import Runtime
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime
from openhands.runtime.impl.docker.docker_runtime import (
    DockerRuntime,
)
from openhands.runtime.impl.kubernetes.kubernetes_runtime import KubernetesRuntime
from openhands.runtime.impl.local.local_runtime import LocalRuntime
from openhands.runtime.impl.remote.remote_runtime import RemoteRuntime
from openhands.utils.import_utils import get_impl

# mypy: disable-error-code="type-abstract"
_ALL_RUNTIME_CLASSES: dict[str, type[Runtime]] = {
    'eventstream': DockerRuntime,
    'docker': DockerRuntime,
    'remote': RemoteRuntime,
    'local': LocalRuntime,
    'kubernetes': KubernetesRuntime,
    'cli': CLIRuntime,
}


def get_runtime_cls(name: str) -> type[Runtime]:
    """If name is one of the predefined runtime names (e.g. 'docker'), return its class.
    Otherwise attempt to resolve name as subclass of Runtime and return it.
    Raise on invalid selections.
    """
    if name in _ALL_RUNTIME_CLASSES:
        return _ALL_RUNTIME_CLASSES[name]
    try:
        return get_impl(Runtime, name)
    except Exception as e:
        known_keys = _ALL_RUNTIME_CLASSES.keys()
        raise ValueError(
            f'Runtime {name} not supported, known are: {known_keys}'
        ) from e


# Build __all__ list dynamically based on available runtimes
__all__ = [
    'Runtime',
    'RemoteRuntime',
    'DockerRuntime',
    'KubernetesRuntime',
    'CLIRuntime',
    'LocalRuntime',
    'get_runtime_cls',
]
