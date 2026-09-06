"""Deterministic lexer-based parsers used when tree-sitter is unavailable.

Extracts symbols, imports, definitions, and simple call sites for Python,
JavaScript, TypeScript, and Markdown. Output is a pure function of source.
"""

from __future__ import annotations

import io
import re
import tokenize
from typing import Any

KIND_FILE = "file"
KIND_MODULE = "module"
KIND_FUNCTION = "function"
KIND_CLASS = "class"
KIND_METHOD = "method"
KIND_VARIABLE = "variable"

LANGUAGE_PYTHON = "python"
LANGUAGE_JAVASCRIPT = "javascript"
LANGUAGE_TYPESCRIPT = "typescript"
LANGUAGE_MARKDOWN = "markdown"
SUPPORTED_LANGUAGES = (
    LANGUAGE_PYTHON,
    LANGUAGE_JAVASCRIPT,
    LANGUAGE_TYPESCRIPT,
    LANGUAGE_MARKDOWN,
)

EXTENSION_LANGUAGE = {
    ".py": LANGUAGE_PYTHON,
    ".js": LANGUAGE_JAVASCRIPT,
    ".jsx": LANGUAGE_JAVASCRIPT,
    ".mjs": LANGUAGE_JAVASCRIPT,
    ".cjs": LANGUAGE_JAVASCRIPT,
    ".ts": LANGUAGE_TYPESCRIPT,
    ".tsx": LANGUAGE_TYPESCRIPT,
    ".md": LANGUAGE_MARKDOWN,
    ".markdown": LANGUAGE_MARKDOWN,
}

_JS_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "function",
        "return",
        "typeof",
        "new",
        "void",
        "await",
        "async",
        "class",
        "interface",
        "type",
        "import",
        "export",
        "from",
        "require",
    }
)

_JS_IMPORT_FROM = re.compile(
    r"""import\s+(?:type\s+)?(?:[\w*{}\s,]+)\s+from\s+['"]([^'"]+)['"]"""
)
_JS_IMPORT_SIDE = re.compile(r"""import\s+['"]([^'"]+)['"]""")
_JS_REQUIRE = re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""")
_JS_FUNCTION = re.compile(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)")
_JS_CLASS = re.compile(r"(?:export\s+)?class\s+(\w+)")
_JS_CONST = re.compile(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=")
_JS_METHOD = re.compile(r"(?:async\s+)?(\w+)\s*\([^;{}]*\)\s*\{")
_JS_CALL = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")
_TS_INTERFACE = re.compile(r"(?:export\s+)?interface\s+(\w+)")
_TS_TYPE = re.compile(r"(?:export\s+)?type\s+(\w+)\s*=")
_TS_IDENT = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_CALL = re.compile(r"`([A-Za-z_][\w]*)\(\)`")
_LINE_COMMENT = re.compile(r"//.*?$")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def detect_language(path: str) -> str | None:
    lower = path.lower()
    for ext, language in EXTENSION_LANGUAGE.items():
        if lower.endswith(ext):
            return language
    return None


def empty_result(language: str) -> dict[str, Any]:
    return {
        "language": language,
        "symbols": [],
        "imports": [],
        "calls": [],
        "references": [],
    }


def parse_source(path: str, source: str, language: str) -> dict[str, Any]:
    if language == LANGUAGE_PYTHON:
        return _parse_python(path, source)
    if language == LANGUAGE_JAVASCRIPT:
        return _parse_javascript(path, source, language)
    if language == LANGUAGE_TYPESCRIPT:
        return _parse_javascript(path, source, language)
    if language == LANGUAGE_MARKDOWN:
        return _parse_markdown(path, source)
    return empty_result(language)


def _item(
    name: str,
    kind: str | None = None,
    line: int = 1,
    path: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "start_line": line,
        "end_line": line,
        "path": path,
    }
    if kind is not None:
        row["kind"] = kind
        row["qualified_name"] = name
    if extra:
        row.update(extra)
    return row


def _parse_python(path: str, source: str) -> dict[str, Any]:
    result = empty_result(LANGUAGE_PYTHON)
    if not source:
        return result
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return result
    class_stack: list[tuple[str, int]] = []
    indent = 0
    prev_kind: int | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        kind, value, start = tok.type, tok.string, tok.start[0]
        if kind == tokenize.INDENT:
            indent += 1
        elif kind == tokenize.DEDENT:
            indent = max(0, indent - 1)
            class_stack = [item for item in class_stack if item[1] < indent]
        elif kind == tokenize.NAME:
            if value == "import" and prev_kind != tokenize.NAME:
                i = _python_import(tokens, i, path, result)
                continue
            if value == "from":
                i = _python_from_import(tokens, i, path, result)
                continue
            if value == "class":
                name, nxt = _next_name(tokens, i + 1)
                if name:
                    result["symbols"].append(
                        _item(name, KIND_CLASS, start, path)
                    )
                    class_stack.append((name, indent))
                    i = nxt
                    prev_kind = tokenize.NAME
                    continue
            if value == "def":
                name, nxt = _next_name(tokens, i + 1)
                if name:
                    in_class = bool(class_stack) and indent > class_stack[-1][1]
                    result["symbols"].append(
                        _item(
                            name,
                            KIND_METHOD if in_class else KIND_FUNCTION,
                            start,
                            path,
                        )
                    )
                    i = nxt
                    prev_kind = tokenize.NAME
                    continue
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if (
                nxt
                and nxt.type == tokenize.OP
                and nxt.string == "="
                and indent == 0
                and value not in {"True", "False", "None"}
            ):
                result["symbols"].append(_item(value, KIND_VARIABLE, start, path))
            elif nxt and nxt.type == tokenize.OP and nxt.string == "(":
                if prev_kind != tokenize.NAME or (
                    tokens[i - 1].string not in {"def", "class"}
                ):
                    result["calls"].append(_item(value, line=start, path=path))
            elif value[:1].isupper() and value not in {"True", "False", "None"}:
                result["references"].append(_item(value, line=start, path=path))
        prev_kind = kind
        i += 1
    return result


def _next_name(tokens: list[tokenize.TokenInfo], index: int) -> tuple[str | None, int]:
    while index < len(tokens):
        tok = tokens[index]
        if tok.type == tokenize.NAME:
            return tok.string, index
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            break
        index += 1
    return None, index


def _python_import(
    tokens: list[tokenize.TokenInfo], index: int, path: str, result: dict[str, Any]
) -> int:
    line = tokens[index].start[0]
    index += 1
    while index < len(tokens):
        tok = tokens[index]
        if tok.type == tokenize.NEWLINE:
            break
        if tok.type == tokenize.NAME and tok.string != "as":
            result["imports"].append(
                _item(tok.string.split(".")[0], line=line, path=path)
            )
            while index + 1 < len(tokens) and tokens[index + 1].string == ".":
                index += 2
        index += 1
    return index


def _python_from_import(
    tokens: list[tokenize.TokenInfo], index: int, path: str, result: dict[str, Any]
) -> int:
    line = tokens[index].start[0]
    index += 1
    module_parts: list[str] = []
    while index < len(tokens):
        tok = tokens[index]
        if tok.type == tokenize.NAME and tok.string == "import":
            break
        if tok.type == tokenize.NAME:
            module_parts.append(tok.string)
        elif tok.string not in {".", ","}:
            if tok.type == tokenize.NEWLINE:
                break
        index += 1
    if module_parts:
        result["imports"].append(
            _item(module_parts[0], line=line, path=path, extra={"module": ".".join(module_parts)})
        )
    return index + 1


def _strip_js_comments(source: str) -> str:
    stripped = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), source)
    return "\n".join(_LINE_COMMENT.sub("", line) for line in stripped.splitlines())


def _parse_javascript(path: str, source: str, language: str) -> dict[str, Any]:
    result = empty_result(language)
    if not source:
        return result
    cleaned = _strip_js_comments(source)
    for match in _JS_IMPORT_FROM.finditer(cleaned):
        result["imports"].append(
            _item(match.group(1), line=_line_at(cleaned, match.start()), path=path)
        )
    for match in _JS_IMPORT_SIDE.finditer(cleaned):
        result["imports"].append(
            _item(match.group(1), line=_line_at(cleaned, match.start()), path=path)
        )
    for match in _JS_REQUIRE.finditer(cleaned):
        result["imports"].append(
            _item(match.group(1), line=_line_at(cleaned, match.start()), path=path)
        )
    for match in _JS_FUNCTION.finditer(cleaned):
        result["symbols"].append(
            _item(
                match.group(1),
                KIND_FUNCTION,
                _line_at(cleaned, match.start()),
                path,
            )
        )
    class_spans: list[tuple[int, int, str]] = []
    for match in _JS_CLASS.finditer(cleaned):
        start = match.start()
        line = _line_at(cleaned, start)
        name = match.group(1)
        result["symbols"].append(_item(name, KIND_CLASS, line, path))
        body_start = cleaned.find("{", match.end())
        if body_start >= 0:
            body_end = _matching_brace(cleaned, body_start)
            class_spans.append((body_start, body_end, name))
    seen_methods: set[tuple[str, int]] = set()
    for body_start, body_end, _class_name in class_spans:
        body = cleaned[body_start:body_end]
        for match in _JS_METHOD.finditer(body):
            name = match.group(1)
            if name in _JS_KEYWORDS or name in {"constructor", "if", "for"}:
                continue
            line = _line_at(cleaned, body_start + match.start())
            key = (name, line)
            if key in seen_methods:
                continue
            seen_methods.add(key)
            result["symbols"].append(_item(name, KIND_METHOD, line, path))
    for match in _JS_CONST.finditer(cleaned):
        result["symbols"].append(
            _item(
                match.group(1),
                KIND_VARIABLE,
                _line_at(cleaned, match.start()),
                path,
            )
        )
    if language == LANGUAGE_TYPESCRIPT:
        for match in _TS_INTERFACE.finditer(cleaned):
            result["symbols"].append(
                _item(
                    match.group(1),
                    KIND_CLASS,
                    _line_at(cleaned, match.start()),
                    path,
                )
            )
        for match in _TS_TYPE.finditer(cleaned):
            result["symbols"].append(
                _item(
                    match.group(1),
                    KIND_CLASS,
                    _line_at(cleaned, match.start()),
                    path,
                )
            )
        for match in _TS_IDENT.finditer(cleaned):
            name = match.group(1)
            line = _line_at(cleaned, match.start())
            if any(
                item["name"] == name and item["start_line"] == line
                for item in result["symbols"]
            ):
                continue
            result["references"].append(_item(name, line=line, path=path))
    for match in _JS_CALL.finditer(cleaned):
        name = match.group(1)
        if name in _JS_KEYWORDS:
            continue
        result["calls"].append(
            _item(name, line=_line_at(cleaned, match.start()), path=path)
        )
    return result


def _matching_brace(source: str, start: int) -> int:
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(source)


def _line_at(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def _parse_markdown(path: str, source: str) -> dict[str, Any]:
    result = empty_result(LANGUAGE_MARKDOWN)
    if not source:
        return result
    for line_no, line in enumerate(source.splitlines(), start=1):
        heading = _MD_HEADING.match(line)
        if heading:
            result["symbols"].append(
                _item(heading.group(2).strip(), KIND_VARIABLE, line_no, path)
            )
        for match in _MD_LINK.finditer(line):
            result["imports"].append(_item(match.group(2), line=line_no, path=path))
        for match in _MD_CALL.finditer(line):
            result["calls"].append(_item(match.group(1), line=line_no, path=path))
    return result
