"""Read MCP wire types written under a different ``mcp`` major version.

fastmcp 4 / ``mcp`` 2.x renamed several ``mcp.types`` fields to snake_case
(``Tool.inputSchema`` -> ``input_schema``, ``CallToolResult.isError`` ->
``is_error``, ``ImageContent.mimeType`` -> ``mime_type``) and serialise the new
spelling into persisted events. The 1.x models are strict: they only accept the
camelCase spelling and raise a ``ValidationError`` for anything else, so a
conversation whose ``SystemPromptEvent.tools`` contains an ``MCPToolDefinition``
fails to restore once the agent-server resolves ``mcp`` 1.x (see
OpenHands/OpenHands#17615).

``mcp`` 2.x accepts both spellings, but the agent-server pins ``fastmcp<4``
resolves 1.x, so we make the 1.x models accept the 2.x spelling too. Each model
field gets an ``AliasChoices`` covering its declared name plus the snake_case
and camelCase variants; nothing about the outgoing spelling changes, so events
serialise exactly as before.

The module is imported at agent-server startup (``--import-modules
mcp_legacy_schema_compat``) and is a no-op on ``mcp`` 2.x, which already reads
both spellings. Remove it once the pinned agent-server reads both natively.
"""

import re
import sys

from pydantic import AliasChoices, BaseModel


def _to_snake_case(name: str) -> str:
    with_underscores = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", with_underscores).lower()


def _to_camel_case(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _field_variants(field_name: str, existing) -> set[str]:
    """Every spelling ``field_name`` may have been serialised under."""
    variants = {field_name, _to_snake_case(field_name), _to_camel_case(field_name)}
    if isinstance(existing, str):
        variants.add(existing)
    elif isinstance(existing, AliasChoices):
        variants.update(existing.choices)
    # An empty variant would make every payload match, so drop it.
    return {v for v in variants if v}


def _iter_mcp_models(mcp_types) -> list[type[BaseModel]]:
    seen: list[type[BaseModel]] = []
    for value in vars(mcp_types).values():
        if (
            isinstance(value, type)
            and issubclass(value, BaseModel)
            and value is not BaseModel
            and value not in seen
        ):
            seen.append(value)
    return seen


def _installed_mcp_rejects_snake_case(mcp_types) -> bool:
    """Whether the installed ``mcp`` is the strict 1.x reader that needs help."""
    try:
        mcp_types.Tool.model_validate(
            {"name": "_probe", "input_schema": {"type": "object"}}
        )
    except Exception:
        return True
    return False


def apply_mcp_field_compatibility(mcp_types=None) -> bool:
    """Accept both ``mcp`` 1.x and 2.x field spellings on every wire model.

    Returns ``True`` when aliases were installed, ``False`` when the installed
    ``mcp`` already reads both spellings (2.x) or when patching failed.
    """
    if mcp_types is None:
        try:
            import mcp.types as mcp_types  # type: ignore[no-redef]
        except Exception:  # pragma: no cover - mcp is always importable here
            return False

    if not _installed_mcp_rejects_snake_case(mcp_types):
        return False

    patched: list[type[BaseModel]] = []
    for model in _iter_mcp_models(mcp_types):
        model_patched = False
        for field_name, field in list(model.model_fields.items()):
            variants = _field_variants(field_name, field.validation_alias)
            if len(variants) <= 1:
                continue
            field.validation_alias = AliasChoices(*sorted(variants))
            model_patched = True
        if not model_patched:
            continue
        try:
            model.model_rebuild(force=True)
            patched.append(model)
        except Exception as error:  # pragma: no cover - defensive
            print(f"mcp_legacy_schema_compat: could not rebuild {model.__name__}: {error}")

    if patched:
        _rebuild_dependents(patched)

    return bool(patched)


def _rebuild_dependents(patched_models: list[type[BaseModel]]) -> None:
    """Recompile SDK models that embed a patched ``mcp`` model.

    Pydantic bakes each field's validator into the enclosing model at class
    construction, so recompiling ``mcp.types.Tool`` alone leaves
    ``MCPToolDefinition`` (and therefore every event that carries it) holding
    the pre-patch validator. Importing the SDK's event machinery here would
    register tools as an import side effect, so instead we force-rebuild every
    already-imported ``openhands`` / ``mcp`` model and let the unimported ones
    pick up the patched fields when they are first built.
    """
    already_rebuilt = set(patched_models)
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(("openhands", "mcp")):
            continue
        for value in list(vars(module).values()):
            if (
                isinstance(value, type)
                and issubclass(value, BaseModel)
                and value is not BaseModel
                and value not in already_rebuilt
            ):
                try:
                    value.model_rebuild(force=True)
                except Exception:  # pragma: no cover - defensive
                    pass


# Patch at import: agent-server startup passes this module to --import-modules.
apply_mcp_field_compatibility()
