"""Optional tree-sitter wrapper. Never imported at module load time.

When tree-sitter or a language grammar is missing, parse_source falls back
to parser_fallback. Callers must lazy-import this module.
"""

from __future__ import annotations

from typing import Any

from parser_fallback import parse_source as fallback_parse_source


def parse_source(path: str, source: str, language: str) -> dict[str, Any]:
    parsed = _try_tree_sitter(path, source, language)
    if parsed is not None:
        return parsed
    return fallback_parse_source(path, source, language)


def _try_tree_sitter(path: str, source: str, language: str) -> dict[str, Any] | None:
    try:
        import tree_sitter  # type: ignore
    except Exception:
        return None
    grammar = _load_grammar(language)
    if grammar is None:
        return None
    try:
        parser = tree_sitter.Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(grammar)
        else:
            parser.language = grammar
        tree = parser.parse(source.encode("utf-8"))
        if tree is None or tree.root_node is None:
            return None
        from parser_fallback import empty_result

        result = empty_result(language)
        _walk(tree.root_node, path, language, result)
        return result
    except Exception:
        return None


def _load_grammar(language: str) -> Any:
    module_names = {
        "python": ("tree_sitter_python",),
        "javascript": ("tree_sitter_javascript",),
        "typescript": ("tree_sitter_typescript", "tree_sitter_javascript"),
        "markdown": ("tree_sitter_markdown",),
    }
    for name in module_names.get(language, ()):
        try:
            module = __import__(name)
        except Exception:
            continue
        language_fn = getattr(module, "language", None)
        if language_fn is None:
            continue
        try:
            import tree_sitter  # type: ignore

            return tree_sitter.Language(language_fn())
        except Exception:
            try:
                return language_fn()
            except Exception:
                continue
    return None


def _walk(node: Any, path: str, language: str, result: dict[str, Any]) -> None:
    # ponytail: tree-sitter is a fidelity upgrade; fallback parsers are the
    # tested contract. Walk only if a grammar actually loaded.
    from parser_fallback import KIND_CLASS, KIND_FUNCTION, KIND_METHOD, KIND_VARIABLE, _item

    ntype = getattr(node, "type", "")
    start = getattr(node, "start_point", (0, 0))[0] + 1
    name_node = None
    for child in getattr(node, "children", []) or []:
        if getattr(child, "type", "") in {"identifier", "name", "property_identifier"}:
            name_node = child
            break
    name = ""
    if name_node is not None:
        try:
            name = name_node.text.decode("utf-8")
        except Exception:
            name = ""
    if ntype in {"function_definition", "function_declaration", "method_definition"}:
        kind = KIND_METHOD if ntype == "method_definition" else KIND_FUNCTION
        if name:
            result["symbols"].append(_item(name, kind, start, path))
    elif ntype in {"class_definition", "class_declaration", "interface_declaration", "type_alias_declaration"}:
        if name:
            result["symbols"].append(_item(name, KIND_CLASS, start, path))
    elif ntype in {"import_statement", "import_from_statement", "import_declaration"}:
        spec = _import_spec(node)
        if spec:
            result["imports"].append(_item(spec, line=start, path=path))
    elif ntype in {"call", "call_expression"}:
        if name:
            result["calls"].append(_item(name, line=start, path=path))
    elif ntype in {"assignment", "lexical_declaration"} and name:
        result["symbols"].append(_item(name, KIND_VARIABLE, start, path))
    for child in getattr(node, "children", []) or []:
        _walk(child, path, language, result)


def _import_spec(node: Any) -> str:
    for child in getattr(node, "children", []) or []:
        ntype = getattr(child, "type", "")
        if ntype in {"dotted_name", "dotted_as_name", "string", "string_fragment"}:
            try:
                text = child.text.decode("utf-8").strip("'\"")
            except Exception:
                continue
            if text:
                return text.split(".")[0]
        nested = _import_spec(child)
        if nested:
            return nested
    return ""
