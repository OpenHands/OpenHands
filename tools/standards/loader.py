"""Import-time-safe plugin loading. Broken modules never crash the registry."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Any

from standards import PluginLoadError, SOURCE_BUILTIN, StandardsPlugin

PLUGIN_ATTR = "PLUGIN"


def _module_name(path: str, source: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"oh_standards_{source}_{stem}_{abs(hash(path))}"


def _plugin_from_module(module: ModuleType) -> StandardsPlugin | None:
    candidate: Any = getattr(module, PLUGIN_ATTR, None)
    if isinstance(candidate, StandardsPlugin):
        return candidate
    if isinstance(candidate, type) and issubclass(candidate, StandardsPlugin):
        return candidate()
    for value in vars(module).values():
        if isinstance(value, StandardsPlugin):
            return value
        if (
            isinstance(value, type)
            and issubclass(value, StandardsPlugin)
            and value is not StandardsPlugin
        ):
            return value()
    return None


def load_plugin_module(
    path: str,
    source: str = SOURCE_BUILTIN,
) -> tuple[StandardsPlugin | None, PluginLoadError | None]:
    if not path.endswith(".py") or os.path.basename(path).startswith("_"):
        return None, None
    name = _module_name(path, source)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None, PluginLoadError(path=path, message="no loader", source=source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        plugin = _plugin_from_module(module)
        if plugin is None:
            return None, PluginLoadError(
                path=path,
                message="module does not export a StandardsPlugin",
                source=source,
            )
        if not getattr(plugin, "name", None):
            return None, PluginLoadError(
                path=path, message="plugin is missing a name", source=source
            )
        plugin.source = source  # type: ignore[attr-defined]
        return plugin, None
    except Exception as exc:  # noqa: BLE001 — isolation is the contract
        sys.modules.pop(name, None)
        return None, PluginLoadError(path=path, message=str(exc), source=source)


def load_plugins_from_dir(
    directory: str,
    source: str = SOURCE_BUILTIN,
) -> tuple[list[StandardsPlugin], list[PluginLoadError]]:
    plugins: list[StandardsPlugin] = []
    errors: list[PluginLoadError] = []
    if not directory or not os.path.isdir(directory):
        return plugins, errors
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        plugin, error = load_plugin_module(path, source)
        if error is not None:
            errors.append(error)
        elif plugin is not None:
            plugins.append(plugin)
    return plugins, errors
