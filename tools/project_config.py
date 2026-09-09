"""Parse and validate ``.openhands/project.yaml``."""

from __future__ import annotations

import json
import os
import re
from typing import Any

SCHEMA_FILENAME = "project_config_schema.json"
PROJECT_CONFIG_RELATIVE_PATH = os.path.join(".openhands", "project.yaml")
ROUTING_PROVIDERS = ("claude", "cursor", "opencode")
CARD_TYPES = ("task", "bug", "feature", "discussion", "refinement")
CHANNEL_TYPES = ("slack", "whatsapp", "buzz")


class ProjectConfigError(ValueError):
    """Invalid project config."""


def schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), SCHEMA_FILENAME)


def load_schema() -> dict[str, Any]:
    with open(schema_path(), encoding="utf-8") as handle:
        return json.load(handle)


def parse_project_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ProjectConfigError("Project config must be a mapping")
        return loaded
    except ImportError:
        return _parse_simple_yaml(text)


def load_project_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return parse_project_yaml(handle.read())


def validate_project_config(data: dict[str, Any]) -> dict[str, Any]:
    if "project" not in data or not isinstance(data["project"], dict):
        raise ProjectConfigError("Missing project object")
    extra_top = set(data) - {"project"}
    if extra_top:
        raise ProjectConfigError(f"Unknown top-level keys: {sorted(extra_top)}")
    project = data["project"]
    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProjectConfigError("project.name is required")
    _optional_string(project, "description")
    if "routing" in project:
        _validate_routing(project["routing"])
    if "kanban" in project:
        _validate_kanban(project["kanban"])
    if "lint" in project:
        _validate_lint(project["lint"])
    if "channels" in project:
        _validate_channels(project["channels"])
    if "standards" in project:
        _validate_standards(project["standards"])
    if "cost_cap" in project and not _is_number(project["cost_cap"]):
        raise ProjectConfigError("project.cost_cap must be a number")
    if "cost_cap" in project and project["cost_cap"] < 0:
        raise ProjectConfigError("project.cost_cap must be >= 0")
    extra = set(project) - {
        "name",
        "description",
        "routing",
        "kanban",
        "lint",
        "channels",
        "standards",
        "cost_cap",
    }
    if extra:
        raise ProjectConfigError(f"Unknown project keys: {sorted(extra)}")
    load_schema()  # ensure the committed schema stays readable
    return data


def _optional_string(obj: dict[str, Any], key: str) -> None:
    if key in obj and obj[key] is not None and not isinstance(obj[key], str):
        raise ProjectConfigError(f"{key} must be a string")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_routing(routing: Any) -> None:
    if not isinstance(routing, dict):
        raise ProjectConfigError("routing must be an object")
    rules = routing.get("rules", [])
    if not isinstance(rules, list):
        raise ProjectConfigError("routing.rules must be a list")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ProjectConfigError("routing rule must be an object")
        if rule.get("provider") not in ROUTING_PROVIDERS:
            raise ProjectConfigError(
                f"provider must be one of {', '.join(ROUTING_PROVIDERS)}"
            )
        if not rule.get("task_type") or not rule.get("model"):
            raise ProjectConfigError("routing rule needs task_type and model")
    default = routing.get("default")
    if default is not None:
        if not isinstance(default, dict):
            raise ProjectConfigError("routing.default must be an object")
        if default.get("provider") not in ROUTING_PROVIDERS:
            raise ProjectConfigError("routing.default.provider is invalid")
        if not default.get("model"):
            raise ProjectConfigError("routing.default.model is required")


def _validate_kanban(kanban: Any) -> None:
    if not isinstance(kanban, dict):
        raise ProjectConfigError("kanban must be an object")
    columns = kanban.get("columns", [])
    if not isinstance(columns, list):
        raise ProjectConfigError("kanban.columns must be a list")
    for column in columns:
        if not isinstance(column, dict) or not column.get("name"):
            raise ProjectConfigError("kanban column needs a name")
    card_types = kanban.get("card_types", [])
    if not isinstance(card_types, list):
        raise ProjectConfigError("kanban.card_types must be a list")
    for card_type in card_types:
        if card_type not in CARD_TYPES:
            raise ProjectConfigError(f"Unknown card type: {card_type}")


def _validate_lint(lint: Any) -> None:
    if not isinstance(lint, dict) or not isinstance(lint.get("rules", []), list):
        raise ProjectConfigError("lint.rules must be a list of strings")
    if any(not isinstance(rule, str) for rule in lint.get("rules", [])):
        raise ProjectConfigError("lint.rules must be strings")


def _validate_channels(channels: Any) -> None:
    if not isinstance(channels, list):
        raise ProjectConfigError("channels must be a list")
    for channel in channels:
        if not isinstance(channel, dict) or channel.get("type") not in CHANNEL_TYPES:
            raise ProjectConfigError(
                f"channel type must be one of {', '.join(CHANNEL_TYPES)}"
            )
        if not isinstance(channel.get("config"), dict):
            raise ProjectConfigError("channel config must be an object")


def _validate_standards(standards: Any) -> None:
    if not isinstance(standards, dict):
        raise ProjectConfigError("standards must be an object")
    plugins = standards.get("plugins", [])
    if not isinstance(plugins, list):
        raise ProjectConfigError("standards.plugins must be a list")
    for plugin in plugins:
        if not isinstance(plugin, dict) or not plugin.get("name"):
            raise ProjectConfigError("plugin needs a name")
        if not isinstance(plugin.get("enabled"), bool):
            raise ProjectConfigError("plugin.enabled must be a boolean")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Indentation-based subset: mappings, lists, scalars. No anchors."""
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, raw in enumerate(lines):
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ProjectConfigError("List item without a list parent")
            item_text = stripped[2:]
            if ": " in item_text:
                key, rest = item_text.split(": ", 1)
                nested = {key: _parse_scalar(rest)}
                parent.append(nested)
                stack.append((indent, nested))
            elif item_text.endswith(":") and item_text.count(":") == 1:
                nested = {}
                parent.append({item_text[:-1].strip(): nested})
                stack.append((indent, nested))
            else:
                parent.append(_parse_scalar(item_text))
            continue
        if ":" not in stripped:
            raise ProjectConfigError(f"Invalid line: {stripped}")
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        if not isinstance(parent, dict):
            raise ProjectConfigError("Cannot assign key on a list")
        if rest == "":
            nxt = lines[index + 1] if index + 1 < len(lines) else ""
            nxt_indent = len(nxt) - len(nxt.lstrip(" ")) if nxt else indent
            child: Any = (
                []
                if nxt.strip().startswith("- ") and nxt_indent > indent
                else {}
            )
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def _parse_scalar(text: str) -> Any:
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "None", "~"}:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text
