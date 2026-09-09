"""Dispatch-time standards prompt injection.

Appends a ``<STANDARDS_REQUIREMENTS>`` block built from each enabled
plugin's ``prompt_instructions()``. No LLM is invoked here.
"""

from __future__ import annotations

from typing import Any

from standards import STANDARDS_REQUIREMENTS_CLOSE, STANDARDS_REQUIREMENTS_OPEN
from standards.registry import StandardsRegistry, get_active_registry


def compose_standards_block(
    registry: StandardsRegistry | None = None,
    *,
    project_root: str | None = None,
) -> str:
    store = registry or get_active_registry()
    if store is None:
        return ""
    if project_root:
        store.discover(project_root)
    config = store.load_config(project_root)
    if not config.get("enabled", True):
        return ""
    enforcement = config.get("enforcement") or {}
    if not enforcement.get("prompt", True):
        return ""
    sections: list[str] = []
    for item in config.get("plugins") or []:
        if not item.get("enabled"):
            continue
        plugin = store.get(str(item.get("name") or ""))
        if plugin is None:
            continue
        text = (plugin.prompt_instructions() or "").strip()
        if not text:
            continue
        sections.append(f"### {plugin.display_name} ({plugin.name})\n{text}")
    if not sections:
        return ""
    body = "\n\n".join(sections)
    return (
        f"{STANDARDS_REQUIREMENTS_OPEN}\n{body}\n{STANDARDS_REQUIREMENTS_CLOSE}"
    )


def apply_dispatch_standards_context(
    run_spec: dict[str, Any],
    *,
    registry: StandardsRegistry | None = None,
) -> dict[str, Any]:
    spec = dict(run_spec)
    root = spec.get("root") or spec.get("worktree_dir")
    root_s = str(root) if root else None
    block = compose_standards_block(registry, project_root=root_s)
    spec["standards_block"] = block or None
    if not block:
        return spec
    prompt = str(
        spec.get("spec_text") or spec.get("prompt") or spec.get("task_text") or ""
    )
    combined = f"{prompt}\n\n{block}".strip() if prompt else block
    spec["spec_text"] = combined
    spec["prompt"] = combined
    return spec


def standards_block_in_prompt(prompt: str | None) -> bool:
    return STANDARDS_REQUIREMENTS_OPEN in (prompt or "")
