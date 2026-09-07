"""Link agent sessions to kanban cards and drive auto-progress."""

from __future__ import annotations

from typing import Any

from kanban import KanbanError, KanbanStore

IN_PROGRESS_COLUMN = "In Progress"
REVIEW_COLUMN = "Review"
IN_PROGRESS_STATUS = "in_progress"
REVIEW_STATUS = "review"



def link_session(store: KanbanStore, card_id: str, session_id: str) -> dict[str, Any]:
    session_id = (session_id or "").strip()
    if not session_id:
        raise KanbanError("session_id is required")
    card = store.update_card(
        card_id,
        agent_session_id=session_id,
        status=IN_PROGRESS_STATUS,
        assignee=session_id,
    )
    progress = store.column_named(card["board_id"], IN_PROGRESS_COLUMN)
    if card["column_id"] != progress["id"]:
        card = store.move_card(card_id, progress["id"], 0)
    return store.append_activity(card_id, f"Linked agent session {session_id}")


def record_progress(
    store: KanbanStore,
    card_id: str,
    message: str,
    status: str | None = None,
) -> dict[str, Any]:
    message = (message or "").strip()
    if not message:
        raise KanbanError("Progress message is required")
    if status:
        store.update_card(card_id, status=status)
    return store.append_activity(card_id, message)


def complete_session(
    store: KanbanStore,
    card_id: str,
    *,
    tests_passed: bool,
    actual_tokens: int | None = None,
    actual_cost: float | None = None,
    tool_calls: int | None = None,
    agent_time: float | None = None,
    model_used: str | None = None,
) -> dict[str, Any]:
    card = store.update_card(
        card_id,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
        tool_calls=tool_calls,
        agent_time=agent_time,
        model_used=model_used,
        status=REVIEW_STATUS if tests_passed else IN_PROGRESS_STATUS,
    )
    if tests_passed:
        review = store.column_named(card["board_id"], REVIEW_COLUMN)
        if card["column_id"] != review["id"]:
            store.move_card(card_id, review["id"], 0)
        return store.append_activity(
            card_id, "Agent finished; tests passed. Moved to Review."
        )
    return store.append_activity(
        card_id, "Agent finished; tests failed. Left in In Progress."
    )
