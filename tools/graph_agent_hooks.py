"""Dispatch-time graph context hooks and indexer singleton access."""

from __future__ import annotations

import os
from typing import Any

from codebase_graph import GraphStore, get_active_store, set_active_store
from graph_context import GRAPH_CONTEXT_OPEN, attach_graph_context

GRAPH_NOTE_PREFIX = "graph_context:"
GRAPH_CONTEXT_FILENAME = "graph_context.md"


def write_graph_context_file(worktree_dir: str | None, block: str | None) -> None:
    if not block or not worktree_dir or not os.path.isdir(str(worktree_dir)):
        return
    path = os.path.join(str(worktree_dir), GRAPH_CONTEXT_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(block)
            handle.write("\n")
    except OSError:
        pass


def apply_dispatch_graph_context(
    run_spec: dict[str, Any],
    *,
    store: GraphStore | None = None,
    kanban_store: Any | None = None,
) -> dict[str, Any]:
    attached = attach_graph_context(run_spec, store=store)
    context = attached.get("graph_context") or {}
    write_graph_context_file(
        attached.get("worktree_dir"),
        context.get("block"),
    )
    card_id = attached.get("card_id")
    if kanban_store is not None and card_id:
        files = context.get("files") or []
        summary = ", ".join(item.get("file") or "" for item in files[:5])
        skipped = context.get("skipped")
        message = (
            f"{GRAPH_NOTE_PREFIX} skipped stale index"
            if skipped
            else f"{GRAPH_NOTE_PREFIX} {summary or 'none'}"
        )
        try:
            kanban_store.append_activity(str(card_id), message)
        except Exception:
            pass
    return attached


def graph_context_in_prompt(prompt: str | None) -> bool:
    return GRAPH_CONTEXT_OPEN in (prompt or "")
