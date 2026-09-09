"""Project bootstrap: LLM decomposition, deep-scan detection, and kanban seeding."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kanban import DEFAULT_COLUMNS, DEFAULT_CARD_STATUS, KanbanStore

TODO_RE = re.compile(r"\b(TODO|FIXME)\b[:\s]*(.*)$", re.IGNORECASE)
SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "build",
    "dist",
    ".venv",
    "__pycache__",
    ".tmp",
}
REQUIRED_PACKAGE_SCRIPTS = ("test", "lint", "build")
CI_PATHS = (
    ".github/workflows",
    ".gitlab-ci.yml",
    "Jenkinsfile",
)

LlmComplete = Callable[[str], str]


class DecompositionError(ValueError):
    """Raised when LLM output does not satisfy the feature/epic/ticket rules."""


@dataclass
class SuggestedCard:
    title: str
    description: str
    source: str
    acceptance: list[str] = field(default_factory=list)
    priority: str = "P2"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def build_decomposition_prompt(spec: str) -> str:
    return (
        "Decompose this product spec into features, epics, and tickets.\n"
        "Return JSON only with shape "
        '{"features":[{"name":str,"epics":[{"name":str,"tickets":'
        '[{"title":str,"description":str,"acceptance":[str]}]}]}]}.\n'
        "Each feature must have at least one epic. Each epic must have at least "
        "one ticket. Each ticket must include acceptance criteria.\n\n"
        f"SPEC:\n{spec}"
    )


def parse_decomposition(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise DecompositionError("decomposition must be a JSON object")
    return data


def validate_decomposition(data: dict[str, Any]) -> None:
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise DecompositionError("decomposition must include at least one feature")
    for feature in features:
        if not isinstance(feature, dict) or not str(feature.get("name") or "").strip():
            raise DecompositionError("each feature needs a name")
        epics = feature.get("epics")
        if not isinstance(epics, list) or not epics:
            raise DecompositionError(f"feature {feature.get('name')!r} needs epics")
        for epic in epics:
            if not isinstance(epic, dict) or not str(epic.get("name") or "").strip():
                raise DecompositionError("each epic needs a name")
            tickets = epic.get("tickets")
            if not isinstance(tickets, list) or not tickets:
                raise DecompositionError(f"epic {epic.get('name')!r} needs tickets")
            for ticket in tickets:
                if not isinstance(ticket, dict) or not str(ticket.get("title") or "").strip():
                    raise DecompositionError("each ticket needs a title")
                acceptance = ticket.get("acceptance")
                if not isinstance(acceptance, list) or not acceptance:
                    raise DecompositionError(
                        f"ticket {ticket.get('title')!r} needs acceptance criteria"
                    )


def flatten_decomposition(data: dict[str, Any]) -> list[SuggestedCard]:
    cards: list[SuggestedCard] = []
    for feature in data["features"]:
        for epic in feature["epics"]:
            for ticket in epic["tickets"]:
                description = str(ticket.get("description") or "").strip()
                cards.append(
                    SuggestedCard(
                        title=str(ticket["title"]).strip(),
                        description=description,
                        source="decomposition",
                        acceptance=[str(item) for item in ticket["acceptance"]],
                    )
                )
    return cards


def stub_llm_complete(prompt: str) -> str:
    """Deterministic fallback used when no live LLM is configured."""
    spec = prompt.split("SPEC:\n", 1)[-1].strip()
    title = (spec.splitlines()[0] if spec else "Initial feature")[:80]
    return json.dumps(
        {
            "features": [
                {
                    "name": title,
                    "epics": [
                        {
                            "name": title,
                            "tickets": [
                                {
                                    "title": title,
                                    "description": spec,
                                    "acceptance": ["Spec is implemented"],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )


def preview_project(
    root: str | Path,
    spec: str | None,
    llm_complete: LlmComplete | None = None,
) -> list[SuggestedCard]:
    cards: list[SuggestedCard] = []
    if spec and spec.strip():
        cards.extend(decompose_spec(spec, llm_complete or stub_llm_complete))
    cards.extend(detect_project(root))
    return cards


def decompose_spec(spec: str, llm_complete: LlmComplete) -> list[SuggestedCard]:
    if not spec.strip():
        raise DecompositionError("spec must not be empty")
    data = parse_decomposition(llm_complete(build_decomposition_prompt(spec)))
    validate_decomposition(data)
    return flatten_decomposition(data)


def _scan_readme(root: Path) -> list[SuggestedCard]:
    cards: list[SuggestedCard] = []
    for name in ("README.md", "README", "readme.md"):
        path = root / name
        if not path.is_file():
            continue
        for line in _read_text(path).splitlines():
            match = TODO_RE.search(line)
            if match:
                detail = match.group(2).strip() or match.group(1)
                cards.append(
                    SuggestedCard(
                        title=f"README: {detail}",
                        description=line.strip(),
                        source="readme",
                    )
                )
    return cards


def _iter_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                files.append(path)
    return files


def _scan_code_todos(root: Path) -> list[SuggestedCard]:
    cards: list[SuggestedCard] = []
    for path in _iter_source_files(root):
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        for line in _read_text(path).splitlines():
            match = TODO_RE.search(line)
            if match:
                detail = match.group(2).strip() or match.group(1)
                cards.append(
                    SuggestedCard(
                        title=f"{relative}: {detail}",
                        description=line.strip(),
                        source="code_todo",
                    )
                )
    return cards


def _scan_git_hotspots(root: Path, limit: int = 5) -> list[SuggestedCard]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return []
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "log", "--pretty=format:", "--name-only", "-n", "50"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    counts: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        name = line.strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    hot = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [
        SuggestedCard(
            title=f"Review frequent changes in {path}",
            description=f"{path} changed {count} times in recent history.",
            source="git_history",
            priority="P3",
        )
        for path, count in hot
        if count > 1
    ]


def _scan_issues(root: Path) -> list[SuggestedCard]:
    templates = root / ".github" / "ISSUE_TEMPLATE"
    if templates.exists():
        return [
            SuggestedCard(
                title="Import existing tracker issues",
                description="Issue templates found; sync GitHub/GitLab issues into the board.",
                source="issues",
                priority="P3",
            )
        ]
    return []


def _scan_ci(root: Path) -> list[SuggestedCard]:
    has_ci = any((root / path).exists() for path in CI_PATHS)
    if not has_ci:
        return [
            SuggestedCard(
                title="Add CI pipeline",
                description="No GitHub Actions, GitLab CI, or Jenkins config was found.",
                source="ci",
                priority="P1",
            )
        ]
    workflow_dir = root / ".github" / "workflows"
    if workflow_dir.is_dir():
        contents = "\n".join(
            _read_text(path) for path in workflow_dir.glob("*.yml")
        ) + "\n".join(_read_text(path) for path in workflow_dir.glob("*.yaml"))
        if "test" not in contents.lower() and "vitest" not in contents.lower():
            return [
                SuggestedCard(
                    title="Add CI test coverage",
                    description="CI workflows exist but no test job was detected.",
                    source="ci",
                    priority="P1",
                )
            ]
    return []


def _scan_package_json(root: Path) -> list[SuggestedCard]:
    path = root / "package.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError:
        return []
    scripts = data.get("scripts") if isinstance(data, dict) else {}
    if not isinstance(scripts, dict):
        scripts = {}
    missing = [name for name in REQUIRED_PACKAGE_SCRIPTS if name not in scripts]
    if not missing:
        return []
    return [
        SuggestedCard(
            title=f"Add package.json scripts: {', '.join(missing)}",
            description="Required project scripts are missing.",
            source="package_json",
            priority="P2",
        )
    ]


def detect_project(root: str | Path) -> list[SuggestedCard]:
    path = Path(root)
    cards: list[SuggestedCard] = []
    cards.extend(_scan_readme(path))
    cards.extend(_scan_code_todos(path))
    cards.extend(_scan_git_hotspots(path))
    cards.extend(_scan_issues(path))
    cards.extend(_scan_ci(path))
    cards.extend(_scan_package_json(path))
    return cards


def suggested_card_payload(card: SuggestedCard) -> dict[str, Any]:
    acceptance = ""
    if card.acceptance:
        acceptance = "\n\nAcceptance:\n" + "\n".join(f"- {item}" for item in card.acceptance)
    return {
        "title": card.title,
        "description": f"{card.description}{acceptance}".strip(),
        "priority": card.priority,
        "status": DEFAULT_CARD_STATUS,
    }


def seed_board(
    store: KanbanStore,
    cards: list[SuggestedCard],
    *,
    name: str = "Project board",
    project_id: str | None = None,
) -> dict[str, Any]:
    board = store.create_board(name, project_id=project_id)
    backlog = next(
        column
        for column in board["columns"]
        if column["name"] == DEFAULT_COLUMNS[0]["name"]
    )
    created = [store.create_card(backlog["id"], **suggested_card_payload(card)) for card in cards]
    return {"board": store.get_board(board["id"]), "cards": created}


def init_project(
    root: str | Path,
    spec: str | None,
    store: KanbanStore,
    llm_complete: LlmComplete | None = None,
    *,
    board_name: str = "Project board",
) -> dict[str, Any]:
    cards = preview_project(root, spec, llm_complete)
    seeded = seed_board(
        store, cards, name=board_name, project_id=str(root)
    )
    return {
        "suggested": [card.__dict__ for card in cards],
        "board": seeded["board"],
        "cards": seeded["cards"],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="openhands project init")
    parser.add_argument("command", choices=["init"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--spec", default=None, help="Path to a product spec file")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)
    spec_text = Path(args.spec).read_text(encoding="utf-8") if args.spec else None
    store = KanbanStore(db_path=args.db)
    result = init_project(
        args.root,
        spec_text,
        store,
        llm_complete=stub_llm_complete if spec_text else None,
    )
    print(json.dumps({"board_id": result["board"]["id"], "cards": len(result["cards"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
