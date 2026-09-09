"""Discover, configure, and run standards plugins."""

from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any, Callable, Iterable

from kanban import new_id, utc_now
from project_config import PROJECT_CONFIG_RELATIVE_PATH, load_project_config
from standards import (
    ACTION_BLOCK,
    ACTION_WARN,
    ACTIONS,
    PluginLoadError,
    SEVERITIES,
    SOURCE_BUILTIN,
    SOURCE_PROJECT,
    SOURCE_USER,
    STATUS_FAILED,
    STATUS_PASSED,
    StandardsPlugin,
    StandardsRun,
    Violation,
    ViolationSeverity,
)
from standards.audit_store import AuditStore, default_db_path
from standards.loader import load_plugins_from_dir

SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    ".tmp",
    "coverage",
    "vendor",
}
MAX_FILE_BYTES = 1_000_000
DEFAULT_ENFORCEMENT = {"prompt": True, "automated": True, "gates": True}

_active: StandardsRegistry | None = None


def builtin_plugins_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")


def user_plugins_dir() -> str:
    return os.path.join(
        os.path.expanduser("~"), ".openhands", "agent-canvas", "standards"
    )


def project_plugins_dir(project_root: str | None) -> str | None:
    if not project_root:
        return None
    return os.path.join(project_root, ".openhands", "standards")


def set_active_registry(registry: StandardsRegistry | None) -> None:
    global _active
    _active = registry


def get_active_registry() -> StandardsRegistry | None:
    return _active


def start_default_registry() -> StandardsRegistry | None:
    existing = get_active_registry()
    if existing is not None:
        return existing
    try:
        registry = StandardsRegistry(default_db_path())
        registry.discover()
        set_active_registry(registry)
        return registry
    except Exception:
        return None


class StandardsRegistry:
    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        audit: AuditStore | None = None,
        builtin_dir: str | None = None,
        user_dir: str | None = None,
        project_root: str | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.audit = audit or AuditStore(db_path)
        self.builtin_dir = builtin_dir if builtin_dir is not None else builtin_plugins_dir()
        self.user_dir = user_dir if user_dir is not None else user_plugins_dir()
        self.project_root = project_root
        self.clock = clock or utc_now
        self._plugins: dict[str, StandardsPlugin] = {}
        self._sources: dict[str, str] = {}
        self._errors: list[PluginLoadError] = []

    def close(self) -> None:
        self.audit.close()

    def discover(self, project_root: str | None = None) -> list[StandardsPlugin]:
        root = project_root if project_root is not None else self.project_root
        if project_root is not None:
            self.project_root = project_root
        plugins: dict[str, StandardsPlugin] = {}
        sources: dict[str, str] = {}
        errors: list[PluginLoadError] = []
        layers = (
            (self.builtin_dir, SOURCE_BUILTIN),
            (self.user_dir, SOURCE_USER),
            (project_plugins_dir(root), SOURCE_PROJECT),
        )
        for directory, source in layers:
            if not directory:
                continue
            found, load_errors = load_plugins_from_dir(directory, source)
            errors.extend(load_errors)
            for plugin in found:
                plugins[plugin.name] = plugin
                sources[plugin.name] = source
        self._plugins = plugins
        self._sources = sources
        self._errors = errors
        return list(plugins.values())

    def plugins(self) -> list[StandardsPlugin]:
        return list(self._plugins.values())

    def get(self, name: str) -> StandardsPlugin | None:
        return self._plugins.get(name)

    def load_errors(self) -> list[PluginLoadError]:
        return list(self._errors)

    def _project_yaml_plugins(self, project_root: str | None) -> tuple[list[dict[str, Any]], bool]:
        root = project_root or self.project_root
        if not root:
            return [], False
        path = os.path.join(root, PROJECT_CONFIG_RELATIVE_PATH)
        if not os.path.isfile(path):
            return [], False
        try:
            data = load_project_config(path)
        except Exception:
            return [], False
        if not isinstance(data, dict):
            return [], False
        standards = data.get("standards")
        if not isinstance(standards, dict):
            project = data.get("project")
            standards = project.get("standards") if isinstance(project, dict) else None
        if not isinstance(standards, dict):
            return [], True
        plugins = standards.get("plugins") or []
        if not isinstance(plugins, list):
            return [], True
        cleaned: list[dict[str, Any]] = []
        for item in plugins:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            cleaned.append(
                {
                    "name": str(item["name"]),
                    "enabled": bool(item.get("enabled", False)),
                    "action": ACTION_WARN,
                }
            )
        return cleaned, True

    def _default_plugin_entries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": plugin.name,
                "enabled": False,
                "action": ACTION_WARN,
            }
            for plugin in self.plugins()
        ]

    def _merge_plugin_lists(
        self, *layers: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_name: dict[str, dict[str, Any]] = {}
        for layer in layers:
            for item in layer:
                name = str(item.get("name") or "")
                if not name:
                    continue
                current = by_name.get(name, {"name": name, "enabled": False, "action": ACTION_WARN})
                if "enabled" in item:
                    current["enabled"] = bool(item["enabled"])
                action = item.get("action")
                if action in ACTIONS:
                    current["action"] = action
                by_name[name] = current
        known = {plugin.name for plugin in self.plugins()}
        merged = [by_name[name] for name in by_name if name in known or name in by_name]
        # Keep discovered plugins even if yaml named extras; extras stay for round-trip.
        names = {item["name"] for item in merged}
        for plugin in self.plugins():
            if plugin.name not in names:
                merged.append(
                    {"name": plugin.name, "enabled": False, "action": ACTION_WARN}
                )
        return merged

    def load_config(self, project_root: str | None = None) -> dict[str, Any]:
        yaml_plugins, yaml_source = self._project_yaml_plugins(project_root)
        persisted = self.audit.get_persisted_config() or {}
        enforcement = dict(DEFAULT_ENFORCEMENT)
        if isinstance(persisted.get("enforcement"), dict):
            for key in DEFAULT_ENFORCEMENT:
                if key in persisted["enforcement"]:
                    enforcement[key] = bool(persisted["enforcement"][key])
        enabled = True
        if "enabled" in persisted:
            enabled = bool(persisted["enabled"])
        plugins = self._merge_plugin_lists(
            self._default_plugin_entries(),
            yaml_plugins,
            persisted.get("plugins") if isinstance(persisted.get("plugins"), list) else [],
        )
        return {
            "enabled": enabled,
            "enforcement": enforcement,
            "plugins": plugins,
            "project_yaml_source": yaml_source,
        }

    def save_config(self, patch: dict[str, Any], project_root: str | None = None) -> dict[str, Any]:
        current = self.load_config(project_root)
        if "enabled" in patch:
            current["enabled"] = bool(patch["enabled"])
        if isinstance(patch.get("enforcement"), dict):
            for key in DEFAULT_ENFORCEMENT:
                if key in patch["enforcement"]:
                    current["enforcement"][key] = bool(patch["enforcement"][key])
        if isinstance(patch.get("plugins"), list):
            current["plugins"] = self._merge_plugin_lists(
                current["plugins"],
                [
                    item
                    for item in patch["plugins"]
                    if isinstance(item, dict) and item.get("name")
                ],
            )
        persistable = {
            "enabled": current["enabled"],
            "enforcement": current["enforcement"],
            "plugins": current["plugins"],
        }
        self.audit.save_persisted_config(persistable)
        return self.load_config(project_root)

    def _plugin_action(self, config: dict[str, Any], name: str) -> str:
        for item in config.get("plugins") or []:
            if item.get("name") == name:
                action = item.get("action")
                return action if action in ACTIONS else ACTION_WARN
        return ACTION_WARN

    def _enabled_names(self, config: dict[str, Any], enabled_names: list[str] | None) -> list[str]:
        if enabled_names is not None:
            return list(enabled_names)
        if not config.get("enabled", True):
            return []
        return [
            str(item["name"])
            for item in config.get("plugins") or []
            if item.get("enabled") and item.get("name") in self._plugins
        ]

    def iter_text_files(self, root: str) -> list[tuple[str, str]]:
        files: list[tuple[str, str]] = []
        abs_root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(abs_root):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                rel = os.path.relpath(path, abs_root)
                try:
                    if os.path.getsize(path) > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                try:
                    with open(path, "rb") as handle:
                        raw = handle.read()
                    if b"\x00" in raw:
                        continue
                    content = raw.decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                files.append((rel.replace("\\", "/"), content))
        return files

    def run_checks(
        self,
        root: str,
        enabled_names: list[str] | None = None,
        *,
        files: list[str] | None = None,
        persist_audit: bool = True,
        project_root: str | None = None,
    ) -> StandardsRun:
        started_at = self.clock()
        t0 = time.monotonic()
        run_id = new_id()
        config = self.load_config(project_root or root)
        names = self._enabled_names(config, enabled_names)
        abs_root = os.path.abspath(root)
        scanned = self.iter_text_files(abs_root)
        if files is not None:
            wanted = {item.replace("\\", "/") for item in files}
            scanned = [item for item in scanned if item[0] in wanted]
        violations: list[Violation] = []
        for rel, content in scanned:
            for name in names:
                plugin = self._plugins.get(name)
                if plugin is None:
                    continue
                action = self._plugin_action(config, name)
                try:
                    hits = plugin.check(rel, content)
                except Exception:
                    continue
                for hit in hits:
                    violations.append(
                        Violation(
                            plugin_name=hit.plugin_name or name,
                            rule_id=hit.rule_id,
                            severity=(
                                hit.severity
                                if hit.severity in SEVERITIES
                                else ViolationSeverity.WARNING.value
                            ),
                            file=hit.file or rel,
                            line=hit.line,
                            message=hit.message,
                            remediation=hit.remediation,
                            action=action,
                            fixable=hit.fixable,
                        )
                    )
        counts = {
            "files_scanned": len(scanned),
            "violation_count": len(violations),
            "info": sum(1 for item in violations if item.severity == "info"),
            "warning": sum(1 for item in violations if item.severity == "warning"),
            "error": sum(1 for item in violations if item.severity == "error"),
        }
        blocking = [
            item
            for item in violations
            if item.action == ACTION_BLOCK and item.severity == "error"
        ]
        status = STATUS_PASSED
        if blocking and config.get("enforcement", {}).get("gates", True):
            status = STATUS_FAILED
        duration_ms = int((time.monotonic() - t0) * 1000)
        run = StandardsRun(
            run_id=run_id,
            started_at=started_at,
            duration_ms=duration_ms,
            status=status,
            summary=counts,
            violations=violations,
            worktree=abs_root,
            enforcement=deepcopy(config.get("enforcement") or DEFAULT_ENFORCEMENT),
        )
        if persist_audit:
            self.audit.write_run(
                run_id=run_id,
                started_at=started_at,
                worktree=abs_root,
                enforcement=run.enforcement,
                duration_ms=duration_ms,
                status=status,
                summary=counts,
            )
            self.audit.write_violations(run_id, violations)
        return run

    def plugin_summaries(self, project_root: str | None = None) -> list[dict[str, Any]]:
        config = self.load_config(project_root)
        by_name = {item["name"]: item for item in config["plugins"]}
        rows = []
        for plugin in self.plugins():
            entry = by_name.get(plugin.name, {})
            rows.append(
                {
                    "name": plugin.name,
                    "display_name": plugin.display_name,
                    "description": plugin.description,
                    "version": plugin.version,
                    "source": self._sources.get(plugin.name, SOURCE_BUILTIN),
                    "enabled": bool(entry.get("enabled")),
                    "action": entry.get("action") or ACTION_WARN,
                    "severity_default": plugin.severity_default,
                }
            )
        return rows
