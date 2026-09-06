"""Meetily transcript ingest: parse, type, dedup, and land cards on kanban."""

from __future__ import annotations

import csv
import io
import json
import re
from difflib import SequenceMatcher
from typing import Any, Protocol

from project_config import CARD_TYPES

FORMAT_JSON = "json"
FORMAT_MARKDOWN = "markdown"
FORMAT_CSV = "csv"
DEFAULT_DEDUP_THRESHOLD = 0.86

CARD_TYPE_TASK = "task"
CARD_TYPE_BUG = "bug"
CARD_TYPE_FEATURE = "feature"
CARD_TYPE_DISCUSSION = "discussion"
CARD_TYPE_REFINEMENT = "refinement"

TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (CARD_TYPE_BUG, ("bug", "broken", "crash", "error", "fail", "regression")),
    (CARD_TYPE_FEATURE, ("feature request", "feature:", "add support", "new feature")),
    (CARD_TYPE_DISCUSSION, ("discuss", "question:", "should we", "open question")),
    (CARD_TYPE_REFINEMENT, ("refine", "polish", "cleanup", "chore", "nits")),
    (CARD_TYPE_TASK, ("todo", "action:", "follow up", "task:")),
)


class ActionExtractor(Protocol):
    def summarize_and_extract(
        self, utterances: list[dict[str, str]]
    ) -> dict[str, Any] | None: ...


class NullExtractor:
    def summarize_and_extract(
        self, utterances: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        return None


class RouterExtractor:
    """Optional Phase 7 LLM hook. Returns None when no complete() is wired."""

    def __init__(self, complete: Any | None = None) -> None:
        self.complete = complete

    def summarize_and_extract(
        self, utterances: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        if self.complete is None:
            return None
        blob = "\n".join(
            f"{row.get('speaker', '')}: {row.get('text', '')}" for row in utterances
        )
        try:
            payload = self.complete(blob)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None


def detect_format(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return FORMAT_JSON
    if "," in stripped.splitlines()[0] and "speaker" in stripped.splitlines()[0].lower():
        return FORMAT_CSV
    if "|" in stripped.splitlines()[0] and "speaker" in stripped.splitlines()[0].lower():
        return FORMAT_CSV
    return FORMAT_MARKDOWN


def parse_transcript(text: str, fmt: str | None = None) -> list[dict[str, str]]:
    kind = fmt or detect_format(text)
    if kind == FORMAT_JSON:
        return _parse_json(text)
    if kind == FORMAT_CSV:
        return _parse_csv(text)
    return _parse_markdown(text)


def _parse_json(text: str) -> list[dict[str, str]]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("utterances") or data.get("transcript") or data.get("items") or []
    if isinstance(data, str):
        return [{"speaker": "", "time": "", "text": data}]
    rows: list[dict[str, str]] = []
    for item in data:
        if isinstance(item, str):
            rows.append({"speaker": "", "time": "", "text": item})
            continue
        rows.append(
            {
                "speaker": str(item.get("speaker") or item.get("user") or ""),
                "time": str(item.get("time") or item.get("timestamp") or ""),
                "text": str(item.get("text") or item.get("content") or ""),
            }
        )
    return rows


def _parse_csv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for item in reader:
        lower = {str(key).lower(): value for key, value in item.items()}
        rows.append(
            {
                "speaker": str(lower.get("speaker") or lower.get("user") or ""),
                "time": str(lower.get("time") or lower.get("timestamp") or ""),
                "text": str(lower.get("text") or lower.get("content") or ""),
            }
        )
    return rows


_MD_LINE = re.compile(
    r"^(?:\*\*)?(?P<speaker>[^*:\-]+)(?:\*\*)?\s*(?:\((?P<time>[^)]+)\))?\s*[:\-]\s*(?P<text>.+)$"
)


def _parse_markdown(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("- ").strip()
        if not line or line.startswith("#"):
            heading = line.lstrip("# ").strip()
            if heading:
                rows.append({"speaker": "", "time": "", "text": heading})
            continue
        match = _MD_LINE.match(line)
        if match:
            rows.append(
                {
                    "speaker": match.group("speaker").strip(),
                    "time": (match.group("time") or "").strip(),
                    "text": match.group("text").strip(),
                }
            )
        else:
            rows.append({"speaker": "", "time": "", "text": line})
    return rows


def type_text(text: str) -> str:
    lowered = text.lower()
    for card_type, keywords in TYPE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return card_type
    return CARD_TYPE_TASK


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def title_similarity(left: str, right: str) -> float:
    a = normalize_title(left)
    b = normalize_title(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def extract_actions_deterministic(
    utterances: list[dict[str, str]],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for row in utterances:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        card_type = type_text(text)
        if card_type == CARD_TYPE_TASK and not any(
            marker in text.lower()
            for marker in ("todo", "action", "we should", "let's", "please")
        ):
            if len(text.split()) < 4:
                continue
        items.append(
            {
                "title": text[:80],
                "card_type": card_type,
                "text": text,
            }
        )
    return items


def extract_actions(
    utterances: list[dict[str, str]],
    extractor: ActionExtractor | None = None,
) -> dict[str, Any]:
    llm = extractor.summarize_and_extract(utterances) if extractor is not None else None
    if llm and llm.get("items"):
        items = []
        for item in llm["items"]:
            card_type = item.get("card_type") or type_text(str(item.get("title") or ""))
            if card_type not in CARD_TYPES:
                card_type = type_text(str(item.get("title") or item.get("text") or ""))
            items.append(
                {
                    "title": str(item.get("title") or "")[:80],
                    "card_type": card_type,
                    "text": str(item.get("text") or item.get("title") or ""),
                }
            )
        return {
            "summary": str(llm.get("summary") or ""),
            "items": items,
            "source": "llm",
        }
    items = extract_actions_deterministic(utterances)
    summary = items[0]["text"][:160] if items else ""
    return {"summary": summary, "items": items, "source": "deterministic"}


def find_duplicate(
    title: str,
    existing: list[tuple[str, str]],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> tuple[str, str, float] | None:
    best: tuple[str, str, float] | None = None
    for card_id, existing_title in existing:
        score = title_similarity(title, existing_title)
        if score >= threshold and (best is None or score > best[2]):
            best = (card_id, existing_title, score)
    return best


class MeetilyService:
    def __init__(
        self,
        kanban_store: Any | None = None,
        *,
        extractor: ActionExtractor | None = None,
        channel_host: Any | None = None,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ) -> None:
        self.kanban_store = kanban_store
        self.extractor = extractor if extractor is not None else RouterExtractor()
        self.channel_host = channel_host
        self.dedup_threshold = dedup_threshold

    def preview(
        self,
        text: str,
        fmt: str | None = None,
        board_id: str | None = None,
    ) -> dict[str, Any]:
        utterances = parse_transcript(text, fmt)
        extracted = extract_actions(utterances, self.extractor)
        duplicates: list[dict[str, Any]] = []
        if board_id and self.kanban_store is not None:
            existing = self._existing_titles(board_id)
            for item in extracted["items"]:
                dup = find_duplicate(item["title"], existing, self.dedup_threshold)
                if dup:
                    duplicates.append(
                        {
                            "title": item["title"],
                            "card_type": item["card_type"],
                            "existing_card_id": dup[0],
                            "existing_title": dup[1],
                            "score": dup[2],
                        }
                    )
        return {
            "format": fmt or detect_format(text),
            "utterances": utterances,
            "duplicates": duplicates,
            **extracted,
        }

    def ingest(
        self,
        text: str,
        *,
        board_id: str,
        fmt: str | None = None,
        channel_id: str | None = None,
        session_id: str | None = None,
        channel_ref: str | None = None,
        thread_ref: str | None = None,
    ) -> dict[str, Any]:
        if self.kanban_store is None:
            raise ValueError("kanban_store is required to create cards")
        preview = self.preview(text, fmt)
        existing = self._existing_titles(board_id)
        created: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        column_id = self._backlog_column(board_id)
        for item in preview["items"]:
            dup = find_duplicate(item["title"], existing, self.dedup_threshold)
            if dup:
                duplicates.append(
                    {
                        "title": item["title"],
                        "card_type": item["card_type"],
                        "existing_card_id": dup[0],
                        "existing_title": dup[1],
                        "score": dup[2],
                    }
                )
                continue
            card = self.kanban_store.create_card(
                column_id,
                item["title"],
                description=f"card_type: {item['card_type']}\n\n{item['text']}",
            )
            created.append({**card, "card_type": item["card_type"]})
            existing.append((card["id"], card["title"]))
        result = {
            "summary": preview["summary"],
            "source": preview["source"],
            "format": preview["format"],
            "items": preview["items"],
            "created": created,
            "duplicates": duplicates,
            "utterances": preview["utterances"],
        }
        if self.channel_host and channel_id and channel_ref:
            self.channel_host.send(
                channel_id,
                channel_ref,
                f"Ingested meeting: {len(created)} cards, {len(duplicates)} duplicates",
                thread_ref=thread_ref,
            )
        return result

    def _backlog_column(self, board_id: str) -> str:
        board = self.kanban_store.get_board(board_id)
        columns = board.get("columns") or []
        if not columns:
            raise ValueError("Board has no columns")
        return str(columns[0]["id"])

    def _existing_titles(self, board_id: str) -> list[tuple[str, str]]:
        board = self.kanban_store.get_board(board_id)
        rows: list[tuple[str, str]] = []
        for column in board.get("columns") or []:
            for card in column.get("cards") or []:
                rows.append((str(card["id"]), str(card["title"])))
        return rows
