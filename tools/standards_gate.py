"""Commit-loop standards stage: automated checks + optional policy gate."""

from __future__ import annotations

import os
from typing import Any

from loop_runner import STATUS_SKIPPED
from standards import ACTION_BLOCK, SEVERITY_ERROR
from standards.registry import (
    StandardsRegistry,
    get_active_registry,
    start_default_registry,
)


def _registry_for(worktree_dir: str) -> StandardsRegistry | None:
    registry = get_active_registry() or start_default_registry()
    if registry is None:
        return None
    registry.discover(worktree_dir)
    return registry


def _apply_auto_fixes(registry: StandardsRegistry, root: str, run: Any) -> bool:
    changed = False
    seen: set[tuple[str, str]] = set()
    for item in run.violations:
        if not item.fixable:
            continue
        key = (item.plugin_name, item.file)
        if key in seen:
            continue
        seen.add(key)
        plugin = registry.get(item.plugin_name)
        if plugin is None:
            continue
        path = os.path.join(root, item.file)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                original = handle.read()
            updated = plugin.auto_fix(item.file, original)
        except OSError:
            continue
        if not updated or updated == original:
            continue
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(updated)
            changed = True
        except OSError:
            continue
    return changed


def format_standards_report(run: Any) -> str:
    lines = [
        f"standards {run.status}: {run.summary.get('violation_count', 0)} "
        f"violation(s) in {run.summary.get('files_scanned', 0)} file(s)"
    ]
    for item in run.violations:
        loc = f"{item.file}:{item.line}" if item.line is not None else item.file
        lines.append(
            f"{item.severity} {item.action} {item.rule_id} {loc} {item.message}"
        )
    return "\n".join(lines)


def handle_standards_stage(
    run_id: str,  # noqa: ARG001
    stage: dict[str, Any],  # noqa: ARG001
    worktree_dir: str,
    definition: dict[str, Any],  # noqa: ARG001
) -> tuple[bool | str, str]:
    registry = _registry_for(worktree_dir)
    if registry is None:
        return STATUS_SKIPPED, "standards registry unavailable"
    config = registry.load_config(worktree_dir)
    if not config.get("enabled", True) or not (config.get("enforcement") or {}).get(
        "automated", True
    ):
        return STATUS_SKIPPED, "standards automated off"
    enabled = [
        str(item["name"])
        for item in config.get("plugins") or []
        if item.get("enabled") and item.get("name")
    ]
    if not enabled:
        return True, "no enabled standards plugins"
    from commit_loop import changed_files

    files = None
    if worktree_dir:
        try:
            changed = changed_files(worktree_dir)
            files = changed or None
        except Exception:
            files = None
    run = registry.run_checks(
        worktree_dir,
        enabled_names=enabled,
        files=files or None,
        persist_audit=True,
        project_root=worktree_dir,
    )
    report = format_standards_report(run)
    gates_on = bool((config.get("enforcement") or {}).get("gates", True))
    blocking = [
        item
        for item in run.violations
        if item.action == ACTION_BLOCK and item.severity == SEVERITY_ERROR
    ]
    if not gates_on or not blocking:
        return True, report
    if _apply_auto_fixes(registry, worktree_dir, run):
        rerun = registry.run_checks(
            worktree_dir,
            enabled_names=enabled,
            files=files or None,
            persist_audit=True,
            project_root=worktree_dir,
        )
        still = [
            item
            for item in rerun.violations
            if item.action == ACTION_BLOCK and item.severity == SEVERITY_ERROR
        ]
        if not still:
            return True, format_standards_report(rerun)
        return False, format_standards_report(rerun)
    return False, report
