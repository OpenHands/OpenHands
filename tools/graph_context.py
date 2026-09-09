"""Graph-guided relevance ranking and prompt composition.

Pure function of (index, task, seeds, budget). No LLM, no randomness.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

from codebase_graph import (
    DEFAULT_GRAPH_BUDGET_LINES,
    DEFAULT_MAX_CONTEXT_FILES,
    DEFAULT_STALE_AFTER_MINUTES,
    EDGE_CALL,
    EDGE_IMPORT,
    KIND_FILE,
    KIND_MODULE,
    GraphStore,
    get_active_store,
)

GRAPH_CONTEXT_OPEN = "<GRAPH_CONTEXT>"
GRAPH_CONTEXT_CLOSE = "</GRAPH_CONTEXT>"
REASON_SEED = "seed file"
REASON_IMPORTED_BY_SEED = "imported by seed"
REASON_CALLERS = "callers of {symbol}"
REASON_DEFINED = "defined here"
REASON_KEYWORD = "keyword overlap"
DEFAULT_WALK_DEPTH = 2
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
PATH_RE = re.compile(r"\b[\w./-]+\.(?:py|js|jsx|ts|tsx|md)\b")
STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "please",
        "fix",
        "add",
        "the",
        "function",
        "class",
        "file",
        "code",
        "into",
        "into",
        "run",
        "app",
    }
)


def _rel(path: str, root: str) -> str:
    if os.path.isabs(path):
        try:
            return os.path.relpath(path, root).replace("\\", "/")
        except ValueError:
            return path.replace("\\", "/")
    return path.replace("\\", "/")


def _is_stale(store: GraphStore, root: str) -> bool:
    status = store.status(root)
    last = status.get("last_full_index_at")
    if not last:
        return False
    try:
        indexed = datetime.fromisoformat(str(last))
        if indexed.tzinfo is None:
            indexed = indexed.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    config = store.get_config()
    raw = config.get("stale_after_minutes")
    minutes = DEFAULT_STALE_AFTER_MINUTES if raw is None else int(raw)
    if minutes <= 0:
        return True
    age = (datetime.now(timezone.utc) - indexed).total_seconds() / 60
    return age > minutes


def _file_lines(root: str, rel: str, nodes: list[dict[str, Any]]) -> int:
    path = os.path.join(root, rel)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                return max(1, sum(1 for _ in handle))
        except OSError:
            pass
    ends = [int(n.get("end_line") or 0) for n in nodes if n.get("path") == rel]
    return max(ends) if ends else 1


def _extract_signals(task: str) -> tuple[list[str], list[str], list[str]]:
    idents = []
    seen: set[str] = set()
    for match in IDENT_RE.finditer(task or ""):
        token = match.group(0)
        if token.lower() in STOPWORDS or token in seen:
            continue
        seen.add(token)
        idents.append(token)
    paths = PATH_RE.findall(task or "")
    keywords = [token.lower() for token in idents if token.lower() not in STOPWORDS]
    return idents, paths, keywords


def select_relevant_files(
    store: GraphStore,
    root: str,
    task: str,
    seeds: list[str] | None = None,
    budget: dict[str, int] | None = None,
    walk_depth: int = DEFAULT_WALK_DEPTH,
) -> list[dict[str, Any]]:
    root = os.path.abspath(root)
    config = store.get_config()
    max_files = int((budget or {}).get("max_files") or config.get("max_context_files") or DEFAULT_MAX_CONTEXT_FILES)
    budget_lines = int(
        (budget or {}).get("budget_lines")
        or config.get("graph_budget_lines")
        or DEFAULT_GRAPH_BUDGET_LINES
    )
    depth_cap = max(0, int(walk_depth))
    nodes = store.list_nodes(root)
    edges = store.list_edges(root)
    file_nodes = [n for n in nodes if n.get("kind") == KIND_FILE and n.get("path")]
    idents, path_frags, keywords = _extract_signals(task)
    ranked: dict[str, dict[str, Any]] = {}

    def add(rel: str, relevance: int, reason: str, depth: int = 0) -> None:
        rel = rel.replace("\\", "/")
        if not rel or rel.startswith(".."):
            return
        current = ranked.get(rel)
        row = {
            "file": rel,
            "relevance": relevance,
            "reason": reason,
            "depth": depth,
            "id": rel,
        }
        if current is None or relevance > current["relevance"] or (
            relevance == current["relevance"] and rel < current["file"]
        ):
            ranked[rel] = row

    seed_rels = [_rel(item, root) for item in (seeds or [])]
    for rel in seed_rels:
        add(rel, 1000, REASON_SEED, 0)
    for frag in path_frags:
        add(_rel(frag, root), 900, REASON_SEED, 0)

    by_path: dict[str, str] = {}
    for node in file_nodes:
        by_path[str(node["path"])] = node["id"]
        by_path[os.path.splitext(os.path.basename(str(node["path"])))[0]] = node["id"]

    outgoing: dict[str, list[str]] = {}
    incoming_calls: dict[str, list[str]] = {}
    for edge in edges:
        if edge["kind"] == EDGE_IMPORT:
            src = next((n for n in nodes if n["id"] == edge["src_id"]), None)
            dst = next((n for n in nodes if n["id"] == edge["dst_id"]), None)
            if src and dst and src.get("path") and dst.get("path"):
                outgoing.setdefault(src["path"], []).append(dst["path"])
        if edge["kind"] == EDGE_CALL:
            src = next((n for n in nodes if n["id"] == edge["src_id"]), None)
            dst = next((n for n in nodes if n["id"] == edge["dst_id"]), None)
            if src and src.get("path"):
                incoming_calls.setdefault(str(edge.get("name") or ""), []).append(src["path"])
            if dst and dst.get("path") and src and src.get("path"):
                outgoing.setdefault(src["path"], []).append(dst["path"])

    frontier = list(seed_rels)
    seen_walk = set(frontier)
    for depth in range(1, depth_cap + 1):
        nxt: list[str] = []
        for rel in frontier:
            for dest in outgoing.get(rel, []):
                if dest in seen_walk:
                    continue
                seen_walk.add(dest)
                add(dest, 800 - (depth * 100), REASON_IMPORTED_BY_SEED, depth)
                nxt.append(dest)
        frontier = nxt

    for ident in idents:
        for src_path in incoming_calls.get(ident, []):
            add(
                src_path,
                700,
                REASON_CALLERS.format(symbol=ident),
                1,
            )
        for node in nodes:
            if node.get("name") == ident and node.get("kind") not in {KIND_FILE, KIND_MODULE}:
                path = node.get("path")
                if path:
                    add(path, 400, REASON_DEFINED, 0)

    for node in file_nodes:
        path = str(node.get("path") or "")
        blob = f"{path} {os.path.splitext(os.path.basename(path))[0]}".lower()
        if keywords and any(key in blob for key in keywords):
            add(path, 200, REASON_KEYWORD, 0)

    ordered = sorted(
        ranked.values(),
        key=lambda item: (-int(item["relevance"]), str(item["id"])),
    )
    selected: list[dict[str, Any]] = []
    used_lines = 0
    for item in ordered:
        lines = _file_lines(root, item["file"], nodes)
        if len(selected) >= max_files:
            break
        if used_lines + lines > budget_lines and selected:
            break
        if lines > budget_lines and not selected:
            item = {**item, "lines": min(lines, budget_lines)}
            selected.append(item)
            break
        selected.append({**item, "lines": lines})
        used_lines += lines
    return selected


def compose_graph_context_block(
    files: list[dict[str, Any]],
    *,
    stale: bool = False,
) -> str:
    lines = [GRAPH_CONTEXT_OPEN]
    if stale:
        lines.append("Index is stale; paths below may be out of date.")
    lines.append("Pin these files for the task:")
    for item in files:
        lines.append(f"- {item['file']} — {item['reason']}")
    lines.append(GRAPH_CONTEXT_CLOSE)
    return "\n".join(lines)


def suggest_edit_target(
    store: GraphStore,
    task: str,
    file: str,
    root: str,
) -> dict[str, Any] | None:
    root = os.path.abspath(root)
    rel = _rel(file, root)
    idents, _, _ = _extract_signals(task)
    nodes = [
        n
        for n in store.list_nodes(root)
        if n.get("path") == rel and n.get("kind") not in {KIND_FILE, KIND_MODULE}
    ]
    for ident in idents:
        for node in nodes:
            if node.get("name") == ident:
                return {
                    "name": ident,
                    "file": rel,
                    "start_line": int(node.get("start_line") or 1),
                    "end_line": int(node.get("end_line") or node.get("start_line") or 1),
                }
    if nodes:
        node = sorted(nodes, key=lambda item: str(item.get("id") or ""))[0]
        return {
            "name": node.get("name"),
            "file": rel,
            "start_line": int(node.get("start_line") or 1),
            "end_line": int(node.get("end_line") or 1),
        }
    return None


def attach_graph_context(
    run_spec: dict[str, Any],
    store: GraphStore | None = None,
) -> dict[str, Any]:
    spec = dict(run_spec)
    spec["graph_blocked"] = False
    graph_store = store or get_active_store()
    if graph_store is None:
        return spec
    config = graph_store.get_config()
    enabled = spec.get("graph_enabled")
    if enabled is None:
        enabled = bool(config.get("enabled", True))
    if not enabled:
        spec["graph_context"] = None
        return spec
    root = spec.get("root") or spec.get("worktree_dir")
    if not root or not os.path.isdir(str(root)):
        return spec
    root = os.path.abspath(str(root))
    status = graph_store.status(root)
    if not status.get("last_full_index_at") and not status.get("files_indexed"):
        return spec
    stale = _is_stale(graph_store, root)
    if stale and bool(config.get("strict")):
        spec["graph_stale_banner"] = True
        spec["graph_context"] = {"skipped": True, "reason": "stale"}
        return spec
    task = str(spec.get("task_text") or spec.get("spec_text") or spec.get("prompt") or "")
    files = select_relevant_files(
        graph_store,
        root,
        task,
        seeds=spec.get("seeds") or spec.get("seed_files"),
        budget={
            "max_files": int(config.get("max_context_files") or DEFAULT_MAX_CONTEXT_FILES),
            "budget_lines": int(config.get("graph_budget_lines") or DEFAULT_GRAPH_BUDGET_LINES),
        },
    )
    block = compose_graph_context_block(files, stale=stale)
    prompt = str(spec.get("spec_text") or spec.get("prompt") or spec.get("task_text") or "")
    combined = f"{prompt}\n\n{block}".strip() if prompt else block
    spec["spec_text"] = combined
    spec["prompt"] = combined
    spec["task_text"] = combined
    spec["graph_context"] = {
        "files": files,
        "stale": stale,
        "block": block,
    }
    if stale:
        spec["graph_stale_banner"] = True
    return spec
