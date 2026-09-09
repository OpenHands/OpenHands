"""Dispatch-time routing hooks for loops, feature-dev, and commit-loop.

Imported by those modules; does not reimplement them. Resolve is always
traced. Struggle re-resolves skipping the failed target and resumes from a
state file rather than restarting.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from cost_estimator import empirical_pass_rate, record_routing_outcome
from provider_adapters import AdapterError, dispatch
from router import RouterStore

RESUME_RELATIVE_PATH = os.path.join(".openhands", "routing-resume.json")
DECISION_RELATIVE_PATH = os.path.join(".openhands", "routing-decision.json")
TRACE_MARKER = "\n\n--- routing decision ---\n"
STRUGGLE_REASON = "struggle"
ANNOTATION_WORK = re.compile(r"work[_-]?type\s*[:=]\s*([a-z0-9-]+)", re.I)
ANNOTATION_SENS = re.compile(r"sensitivity\s*[:=]\s*([a-z0-9-]+)", re.I)
DEFAULT_STRUGGLE_THRESHOLD = 2


class StruggleTracker:
    def __init__(self) -> None:
        self._counts: dict[str, dict[str, Any]] = {}

    def record_failure(self, run_id: str, stage_type: str, target: str) -> int:
        state = self._counts.get(run_id) or {
            "stage_type": stage_type,
            "target": target,
            "count": 0,
        }
        if state["stage_type"] == stage_type and state["target"] == target:
            state["count"] = int(state["count"]) + 1
        else:
            state = {"stage_type": stage_type, "target": target, "count": 1}
        self._counts[run_id] = state
        return int(state["count"])

    def reset(self, run_id: str | None = None) -> None:
        if run_id is None:
            self._counts.clear()
            return
        self._counts.pop(run_id, None)


_struggle = StruggleTracker()


def reset_struggle(run_id: str | None = None) -> None:
    _struggle.reset(run_id)


def parse_annotations(text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    work = ANNOTATION_WORK.search(text or "")
    sens = ANNOTATION_SENS.search(text or "")
    if work:
        tags["work_type"] = work.group(1).lower()
    if sens:
        tags["sensitivity"] = sens.group(1).lower()
    return tags


def resume_path(worktree_dir: str) -> str:
    return os.path.join(worktree_dir, RESUME_RELATIVE_PATH)


def write_resume_state(worktree_dir: str, payload: dict[str, Any]) -> str:
    path = resume_path(worktree_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def read_resume_state(worktree_dir: str) -> dict[str, Any] | None:
    path = resume_path(worktree_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def build_resume_prompt(
    ticket_text: str,
    state: dict[str, Any] | None,
) -> str:
    if not state:
        return ticket_text
    branch = state.get("branch") or ""
    workspace = state.get("workspace") or ""
    failed = state.get("failed_output") or ""
    original = state.get("ticket") or ticket_text
    return (
        f"{ticket_text}\n\n"
        "## Resume context (do not restart)\n"
        f"- branch: {branch}\n"
        f"- workspace: {workspace}\n"
        f"- original ticket: {original}\n"
        f"- failed stage output:\n{failed}\n"
    )


def persist_dispatch_trace(
    result: dict[str, Any],
    *,
    worktree_dir: str | None = None,
    kanban_store: Any | None = None,
    card_id: str | None = None,
) -> None:
    decision = result.get("decision") or {}
    trace = result.get("trace") or {}
    payload = {"decision": decision, "trace": trace}
    if worktree_dir:
        path = os.path.join(worktree_dir, DECISION_RELATIVE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    if kanban_store is None or not card_id:
        return
    try:
        card = kanban_store.get_card(card_id)
    except Exception:
        return
    line = (
        f"{TRACE_MARKER}"
        f"{decision.get('provider_key')}/{decision.get('model')}: "
        f"{trace.get('reason') or ''}"
    )
    fields: dict[str, Any] = {
        "description": (card.get("description") or "") + line,
    }
    if decision.get("model"):
        fields["model_used"] = decision["model"]
    kanban_store.update_card(card_id, **fields)


def _empirical_map(work_type: str, models: list[dict[str, Any]]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for model in models:
        rate = empirical_pass_rate(work_type, model["provider_key"], model["id"])
        if rate is not None:
            rates[f"{work_type}:{model['provider_key']}:{model['id']}"] = rate
    return rates


def resolve_for_dispatch(
    store: RouterStore,
    *,
    task_text: str,
    work_type: str | None = None,
    sensitivity: str | None = None,
    connected_providers: list[str] | None = None,
    skip_targets: list[dict[str, Any]] | None = None,
    card_id: str | None = None,
    run_id: str | None = None,
    local_runtimes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tags = parse_annotations(task_text)
    request: dict[str, Any] = {
        "task_text": task_text,
        "work_type": work_type or tags.get("work_type"),
        "sensitivity": sensitivity or tags.get("sensitivity"),
        "card_id": card_id,
        "run_id": run_id,
        "skip_targets": skip_targets or [],
    }
    if connected_providers is not None:
        request["connected_providers"] = connected_providers
    if local_runtimes is not None:
        request["local_runtimes"] = local_runtimes
    config = store.get_config()
    if float(config.get("blend_alpha") or 0) > 0:
        request["empirical_pass_rates"] = _empirical_map(
            request.get("work_type") or "coding",
            store.registry.merged_models(),
        )
    return store.resolve(request)


def dispatch_resolved(
    result: dict[str, Any],
    prompt: str,
    *,
    cwd: str | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    decision = result.get("decision") or {}
    if not decision.get("usable") or not decision.get("provider_key"):
        raise AdapterError("no usable routing target", unusable=True)
    return dispatch(
        str(decision["provider_key"]),
        prompt,
        decision.get("model"),
        cwd=cwd,
        runner=runner,
    )


def escalate_on_struggle(
    store: RouterStore,
    *,
    task_text: str,
    failed_result: dict[str, Any],
    failed_output: str,
    worktree_dir: str,
    branch: str | None = None,
    ticket: str | None = None,
    run_id: str | None = None,
    card_id: str | None = None,
    connected_providers: list[str] | None = None,
    local_runtimes: dict[str, Any] | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
    stage_type: str = "implement",
) -> dict[str, Any]:
    decision = failed_result.get("decision") or {}
    target = f"{decision.get('provider_key')}/{decision.get('model')}"
    count = _struggle.record_failure(run_id or "anon", stage_type, target)
    threshold = int(
        store.get_config().get("struggle_threshold") or DEFAULT_STRUGGLE_THRESHOLD
    )
    if count < threshold:
        return failed_result
    skip = [
        {
            "provider_key": decision.get("provider_key"),
            "model": decision.get("model"),
        }
    ]
    nxt = resolve_for_dispatch(
        store,
        task_text=task_text,
        skip_targets=skip,
        card_id=card_id,
        run_id=run_id,
        connected_providers=connected_providers,
        local_runtimes=local_runtimes,
    )
    store.record_switch(
        from_target={
            "provider_key": decision.get("provider_key"),
            "model": decision.get("model"),
        },
        to_target={
            "provider_key": (nxt.get("decision") or {}).get("provider_key"),
            "model": (nxt.get("decision") or {}).get("model"),
        },
        reason=STRUGGLE_REASON,
        card_id=card_id,
        run_id=run_id,
    )
    state = {
        "branch": branch or "",
        "workspace": worktree_dir,
        "failed_output": failed_output,
        "ticket": ticket or task_text,
        "from": target,
        "to": (
            f"{(nxt.get('decision') or {}).get('provider_key')}/"
            f"{(nxt.get('decision') or {}).get('model')}"
        ),
    }
    write_resume_state(worktree_dir, state)
    prompt = build_resume_prompt(task_text, state)
    nxt["switched"] = True
    nxt["resume_prompt"] = prompt
    persist_dispatch_trace(
        nxt, worktree_dir=worktree_dir, card_id=card_id
    )
    if runner is not None:
        nxt["dispatch"] = dispatch_resolved(
            nxt, prompt, cwd=worktree_dir, runner=runner
        )
    return nxt


def record_outcome(
    work_type: str,
    provider: str,
    model: str,
    passed: bool,
) -> None:
    record_routing_outcome(work_type, provider, model, passed)
