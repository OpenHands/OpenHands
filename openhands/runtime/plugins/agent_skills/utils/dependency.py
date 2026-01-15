# DEPRECATED: This module is part of the deprecated 'openhands.runtime' package.
# It will be removed on April 1, 2025. Please migrate to the OpenHands SDK:
# https://github.com/All-Hands-AI/openhands-sdk
from types import ModuleType


def import_functions(
    module: ModuleType, function_names: list[str], target_globals: dict[str, object]
) -> None:
    for name in function_names:
        if hasattr(module, name):
            target_globals[name] = getattr(module, name)
        else:
            raise ValueError(f'Function {name} not found in {module.__name__}')
