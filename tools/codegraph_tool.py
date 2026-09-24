"""CodeGraph — a persistent, incremental symbol index per project, backed by
SQLite, so the agent can jump straight to a definition/reference instead of
re-scanning the whole tree with grep/glob every time.

Registered as tool name "codegraph". Import at agent-server startup with
``--import-modules codegraph_tool`` (same convention as ``canvas_ui_tool``).

Design:
- One SQLite DB per project, at ``<working_dir>/.openhands/codegraph.db``.
  Each remote machine can host multiple projects; since the DB lives inside
  the project's own working_dir, every project gets its own isolated graph —
  there is no cross-project global store to keep straight.
- Every call refreshes the index incrementally first (mtime+size diff against
  the last scan), then answers the query. No separate "build" step for the
  user to remember.
- File discovery uses `rg --files` when ripgrep is installed (respects
  .gitignore automatically); falls back to os.walk with a conservative
  exclude list otherwise.
- Symbol extraction: precise (ast) for Python; regex heuristics for
  JS/TS/JSX/TSX and PHP (this codebase's two other common stacks); a generic
  regex heuristic for other extensions. Anything not recognized still gets
  tracked as a file (so "references" search still covers it) but contributes
  no symbols.
- "references" shells out to ripgrep (or falls back to the same os.walk
  scan) for a fixed-string search of the symbol name — this is still a scan,
  but scoped to indexed source files and skipping heavy dirs, which is the
  actual cost of the "onerosa" full grep this tool replaces for symbol work.
- Tool description tells the agent explicitly: try codegraph first for
  symbol lookups; grep/glob remain the fallback for full-text search or
  anything codegraph comes up empty on.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import sqlite3
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from pydantic import Field

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import ToolAnnotations, ToolExecutor, register_tool


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


# ---- excludes ---------------------------------------------------------

EXCLUDE_DIRS = {
    ".git", "node_modules", "vendor", "venv", ".venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target", ".cache",
    "coverage", ".tox", ".idea", ".vscode", ".openhands",
}
MAX_FILE_BYTES = 1_500_000  # skip parsing (but still track) huge files
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".php",
    ".go", ".rb", ".java", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".rs", ".kt", ".swift", ".vue", ".scala", ".m",
}


def _db_path(working_dir: str) -> Path:
    d = Path(working_dir) / ".openhands"
    d.mkdir(parents=True, exist_ok=True)
    return d / "codegraph.db"


def _connect(working_dir: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path(working_dir)))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            file TEXT NOT NULL,
            line INTEGER NOT NULL,
            end_line INTEGER,
            parent TEXT,
            signature TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
        CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
            name, sym_id UNINDEXED, tokenize='trigram'
        );
        """
    )
    return conn


# ---- file discovery -----------------------------------------------------


def _list_files(working_dir: str) -> list[str]:
    rg = shutil.which("rg")
    if rg:
        try:
            out = subprocess.run(
                [rg, "--files", "--hidden", "--glob", "!.git/**"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if out.returncode in (0, 1):  # 1 = no matches, still fine
                return [
                    line for line in out.stdout.splitlines() if line.strip()
                ]
        except Exception:
            pass  # fall through to os.walk

    results: list[str] = []
    for root, dirs, files in os.walk(working_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            full = os.path.join(root, f)
            results.append(os.path.relpath(full, working_dir))
    return results


# ---- symbol extraction ---------------------------------------------------


def _extract_python(text: str) -> list[tuple[str, str, int, int | None, str | None, str | None]]:
    """Return (name, kind, line, end_line, parent, signature) tuples."""
    out: list[tuple[str, str, int, int | None, str | None, str | None]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out

    def visit(node: ast.AST, parent: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = ", ".join(a.arg for a in child.args.args)
                sig = f"def {child.name}({args})"
                kind = "method" if parent else "function"
                out.append(
                    (child.name, kind, child.lineno, child.end_lineno, parent, sig)
                )
                visit(child, f"{parent}.{child.name}" if parent else child.name)
            elif isinstance(child, ast.ClassDef):
                out.append(
                    (child.name, "class", child.lineno, child.end_lineno, parent, f"class {child.name}")
                )
                visit(child, child.name)
            else:
                visit(child, parent)

    visit(tree, None)
    return out


_JS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?function\s*\*?\s+([A-Za-z_$][\w$]*)\s*\(",
    re.MULTILINE,
)
_JS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?"
    r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
    re.MULTILINE,
)
_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?class\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_JS_METHOD_RE = re.compile(
    r"^\s{2,}(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*"
    r"([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{",
    re.MULTILINE,
)

_PHP_FUNC_RE = re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|abstract\s+|final\s+)*"
    r"function\s*&?\s*([A-Za-z_][\w]*)\s*\(",
    re.MULTILINE,
)
_PHP_CLASS_RE = re.compile(
    r"^\s*(?:abstract\s+|final\s+)?class\s+([A-Za-z_][\w]*)"
    r"|^\s*interface\s+([A-Za-z_][\w]*)"
    r"|^\s*trait\s+([A-Za-z_][\w]*)",
    re.MULTILINE,
)

_GENERIC_FUNC_RE = re.compile(
    r"^\s*(?:func|def|fn|sub)\s+([A-Za-z_][\w]*)\s*\(",
    re.MULTILINE,
)
_GENERIC_CLASS_RE = re.compile(
    r"^\s*class\s+([A-Za-z_][\w]*)",
    re.MULTILINE,
)


def _lineno(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _extract_js(text: str) -> list[tuple[str, str, int, None, None, None]]:
    out: list[tuple[str, str, int, None, None, None]] = []
    for m in _JS_FUNC_RE.finditer(text):
        out.append((m.group(1), "function", _lineno(text, m.start()), None, None, None))
    for m in _JS_ARROW_RE.finditer(text):
        out.append((m.group(1), "function", _lineno(text, m.start()), None, None, None))
    for m in _JS_CLASS_RE.finditer(text):
        out.append((m.group(1), "class", _lineno(text, m.start()), None, None, None))
    for m in _JS_METHOD_RE.finditer(text):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "catch", "function"):
            continue
        out.append((name, "method", _lineno(text, m.start()), None, None, None))
    return out


def _extract_php(text: str) -> list[tuple[str, str, int, None, None, None]]:
    out: list[tuple[str, str, int, None, None, None]] = []
    for m in _PHP_FUNC_RE.finditer(text):
        out.append((m.group(1), "function", _lineno(text, m.start()), None, None, None))
    for m in _PHP_CLASS_RE.finditer(text):
        name = m.group(1) or m.group(2) or m.group(3)
        out.append((name, "class", _lineno(text, m.start()), None, None, None))
    return out


def _extract_generic(text: str) -> list[tuple[str, str, int, None, None, None]]:
    out: list[tuple[str, str, int, None, None, None]] = []
    for m in _GENERIC_FUNC_RE.finditer(text):
        out.append((m.group(1), "function", _lineno(text, m.start()), None, None, None))
    for m in _GENERIC_CLASS_RE.finditer(text):
        out.append((m.group(1), "class", _lineno(text, m.start()), None, None, None))
    return out


def _extract(path: str, text: str) -> list:
    ext = Path(path).suffix.lower()
    if ext == ".py":
        return _extract_python(text)
    if ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue"):
        return _extract_js(text)
    if ext == ".php":
        return _extract_php(text)
    if ext in CODE_EXTENSIONS:
        return _extract_generic(text)
    return []


# ---- incremental index refresh -------------------------------------------


def _refresh(conn: sqlite3.Connection, working_dir: str) -> dict:
    """Incrementally re-scan the project; returns stats about what changed."""
    disk_files = _list_files(working_dir)
    disk_set = set(disk_files)

    known = dict(conn.execute("SELECT path, mtime FROM files"))
    changed = 0
    removed = 0

    for rel in disk_files:
        full = os.path.join(working_dir, rel)
        try:
            st = os.stat(full)
        except OSError:
            continue
        prev_mtime = known.get(rel)
        if prev_mtime is not None and abs(prev_mtime - st.st_mtime) < 1e-6:
            continue  # unchanged

        changed += 1
        conn.execute("DELETE FROM symbols WHERE file = ?", (rel,))
        ext = Path(rel).suffix.lower()
        if ext in CODE_EXTENSIONS and st.st_size <= MAX_FILE_BYTES:
            try:
                text = Path(full).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            for name, kind, line, end_line, parent, sig in _extract(rel, text):
                conn.execute(
                    "INSERT INTO symbols (name, kind, file, line, end_line, parent, signature)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (name, kind, rel, line, end_line, parent, sig),
                )
        conn.execute(
            "INSERT INTO files (path, mtime, size) VALUES (?,?,?)"
            " ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size",
            (rel, st.st_mtime, st.st_size),
        )

    stale = set(known) - disk_set
    for rel in stale:
        conn.execute("DELETE FROM files WHERE path = ?", (rel,))
        conn.execute("DELETE FROM symbols WHERE file = ?", (rel,))
        removed += 1

    conn.execute("DELETE FROM symbols_fts")
    conn.execute(
        "INSERT INTO symbols_fts(name, sym_id) SELECT name, id FROM symbols"
    )
    conn.commit()
    return {"files_scanned": len(disk_files), "files_changed": changed, "files_removed": removed}


# ---- tool surface ---------------------------------------------------------

CodeGraphCommand = Literal["search", "definition", "references", "status", "rebuild"]


class CodeGraphAction(Action):
    """Query the project's persistent symbol index (SQLite-backed)."""

    command: CodeGraphCommand = Field(description="Operation to perform.")
    symbol: str | None = Field(
        default=None,
        description=(
            "Symbol name (function/class/method). Required for 'search', "
            "'definition', and 'references'. For 'search', substring match; "
            "for 'definition'/'references', exact name."
        ),
    )
    limit: int = Field(default=30, description="Max results to return.")


class CodeGraphObservation(Observation):
    """Result of a codegraph query."""


class CodeGraphExecutor(ToolExecutor[CodeGraphAction, CodeGraphObservation]):
    def __init__(self, working_dir: str):
        self.working_dir = working_dir

    def __call__(
        self,
        action: CodeGraphAction,
        conversation=None,  # noqa: ARG002
    ) -> CodeGraphObservation:
        conn = _connect(self.working_dir)
        try:
            if action.command == "rebuild":
                conn.execute("DELETE FROM files")
                conn.execute("DELETE FROM symbols")
                conn.commit()

            t0 = time.monotonic()
            stats = _refresh(conn, self.working_dir)
            elapsed = time.monotonic() - t0

            if action.command in ("rebuild", "status"):
                n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                n_symbols = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
                return CodeGraphObservation.from_text(
                    f"CodeGraph @ {self.working_dir}\n"
                    f"files indexed: {n_files} | symbols: {n_symbols}\n"
                    f"last refresh: scanned {stats['files_scanned']}, "
                    f"changed {stats['files_changed']}, removed {stats['files_removed']} "
                    f"({elapsed:.2f}s)"
                )

            if action.command == "search":
                if not action.symbol:
                    return CodeGraphObservation.from_text(
                        "Missing 'symbol' for search."
                    )
                rows = conn.execute(
                    """
                    SELECT s.name, s.kind, s.file, s.line, s.parent
                    FROM symbols_fts f
                    JOIN symbols s ON s.id = f.sym_id
                    WHERE symbols_fts MATCH ?
                    ORDER BY s.name
                    LIMIT ?
                    """,
                    (action.symbol, action.limit),
                ).fetchall()
                if not rows:
                    # trigram FTS needs >=3 chars; fall back to LIKE for short queries
                    rows = conn.execute(
                        "SELECT name, kind, file, line, parent FROM symbols"
                        " WHERE name LIKE ? ORDER BY name LIMIT ?",
                        (f"%{action.symbol}%", action.limit),
                    ).fetchall()
                if not rows:
                    return CodeGraphObservation.from_text(
                        f"No symbols matching '{action.symbol}' in the codegraph index. "
                        "Fall back to the grep/glob tools for a full-text search "
                        "(the symbol may be dynamically generated, in an unindexed "
                        "file type, or the name may be different)."
                    )
                lines = [
                    f"{k}: {n}" + (f" (in {p})" if p else "") + f" — {f}:{ln}"
                    for n, k, f, ln, p in rows
                ]
                return CodeGraphObservation.from_text("\n".join(lines))

            if action.command == "definition":
                if not action.symbol:
                    return CodeGraphObservation.from_text(
                        "Missing 'symbol' for definition."
                    )
                rows = conn.execute(
                    "SELECT name, kind, file, line, end_line, parent, signature"
                    " FROM symbols WHERE name = ? LIMIT ?",
                    (action.symbol, action.limit),
                ).fetchall()
                if not rows:
                    return CodeGraphObservation.from_text(
                        f"No exact definition of '{action.symbol}' in the codegraph "
                        "index. Try 'search' for near matches, or fall back to grep."
                    )
                lines = []
                for n, k, f, ln, end_ln, p, sig in rows:
                    loc = f"{f}:{ln}" + (f"-{end_ln}" if end_ln else "")
                    extra = f" — {sig}" if sig else ""
                    parent_s = f" (in {p})" if p else ""
                    lines.append(f"{k} {n}{parent_s} @ {loc}{extra}")
                return CodeGraphObservation.from_text("\n".join(lines))

            if action.command == "references":
                if not action.symbol:
                    return CodeGraphObservation.from_text(
                        "Missing 'symbol' for references."
                    )
                matches = _search_references(self.working_dir, action.symbol, action.limit)
                if not matches:
                    return CodeGraphObservation.from_text(
                        f"No references to '{action.symbol}' found. Fall back to "
                        "grep if you expect matches in excluded paths (e.g. "
                        "node_modules, vendor)."
                    )
                return CodeGraphObservation.from_text("\n".join(matches))

            return CodeGraphObservation.from_text(f"Unknown command: {action.command}")
        finally:
            conn.close()


def _search_references(working_dir: str, symbol: str, limit: int) -> list[str]:
    rg = shutil.which("rg")
    if rg:
        try:
            out = subprocess.run(
                [rg, "--line-number", "--fixed-strings", "--hidden",
                 "--glob", "!.git/**", "--max-count", "500", symbol],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            lines = [line for line in out.stdout.splitlines() if line.strip()]
            return lines[:limit]
        except Exception:
            pass

    results: list[str] = []
    for root, dirs, files in os.walk(working_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fname in files:
            if Path(fname).suffix.lower() not in CODE_EXTENSIONS:
                continue
            full = os.path.join(root, fname)
            try:
                with open(full, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if symbol in line:
                            rel = os.path.relpath(full, working_dir)
                            results.append(f"{rel}:{i}:{line.rstrip()}")
                            if len(results) >= limit:
                                return results
            except OSError:
                continue
    return results


_CODEGRAPH_DESCRIPTION = """Query this project's persistent, SQLite-backed symbol index — a lightweight code graph of function/class/method definitions kept up to date incrementally on every call (no manual build step).

Prefer this tool FIRST whenever you need to:
* find where a function/class/method is DEFINED → command="definition", symbol="<exact name>"
* find symbols by partial name → command="search", symbol="<substring>"
* find where a symbol is USED/called elsewhere → command="references", symbol="<exact name>"
* check index size/health → command="status"
* force a full reindex if results look stale → command="rebuild"

Only fall back to the grep/glob tools when:
* codegraph returns no results and you suspect the symbol is dynamically
  generated, string-built, or lives in a file type the index doesn't parse,
* you need full-text search unrelated to a symbol name (config values,
  comments, log strings), or
* you need to search inside excluded paths (node_modules, vendor, .git).

The index lives at <project>/.openhands/codegraph.db, scoped to the current
project only — each project (including on machines hosting several projects)
gets its own isolated graph, never mixed with another project's symbols."""


class CodeGraphTool(ToolDefinition[CodeGraphAction, CodeGraphObservation]):
    """Tool for querying the project's SQLite-backed code graph."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState" = None,  # noqa: ARG003
        **params,  # noqa: ARG003
    ) -> Sequence["CodeGraphTool"]:
        working_dir = conv_state.workspace.working_dir if conv_state else "."
        return [
            cls(
                description=_CODEGRAPH_DESCRIPTION,
                action_type=CodeGraphAction,
                observation_type=CodeGraphObservation,
                executor=CodeGraphExecutor(working_dir),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool("codegraph", CodeGraphTool)


def _install_default_tools_patch() -> None:
    """Make "codegraph" part of the agent's default tool set.

    Agent profiles don't carry their own `tools` list (the SDK's
    `OpenHandsAgentProfile` has no such field). The actual "single defaulting
    point" (its own docstring's words) for a conversation whose
    `agent_settings.tools` is None is `AgentSettings.create_agent()` in
    `openhands.sdk.settings.model`, which does a *local* `from
    openhands.sdk.tool.defaults import default_tool_specs` inside the method
    body on every call — so rebinding the attribute on the `defaults` module
    before any conversation is created is enough to reach every conversation,
    regardless of which agent profile started it or which higher-level
    wrapper (conversation_router.py's own `get_default_tools` convenience,
    used by a narrower quick-start path) was involved. Patched here instead
    of touching SDK package files, so it survives SDK version bumps as long
    as this import path stays the same.
    """
    import openhands.sdk.tool.defaults as _defaults

    if getattr(_defaults.default_tool_specs, "_codegraph_patched", False):
        return

    _original_default_tool_specs = _defaults.default_tool_specs

    def _default_tool_specs_with_codegraph(*args, **kwargs):
        from openhands.sdk.tool import Tool

        specs = list(_original_default_tool_specs(*args, **kwargs))
        if not any(t.name == "codegraph" for t in specs):
            specs.append(Tool(name="codegraph"))
        return specs

    _default_tool_specs_with_codegraph._codegraph_patched = True  # type: ignore[attr-defined]
    _defaults.default_tool_specs = _default_tool_specs_with_codegraph

    # Also patch the higher-level openhands.tools.preset.default wrapper
    # (used by conversation_router.py's quick-start path) and the module's
    # already-bound `from ... import get_default_tools` in that router, for
    # any conversation-creation path that goes through it instead.
    try:
        import openhands.tools.preset.default as _preset

        _original_get_default_tools = _preset.get_default_tools

        def _get_default_tools_with_codegraph(*args, **kwargs):
            from openhands.sdk.tool import Tool

            tools = list(_original_get_default_tools(*args, **kwargs))
            if not any(t.name == "codegraph" for t in tools):
                tools.append(Tool(name="codegraph"))
            return tools

        _preset.get_default_tools = _get_default_tools_with_codegraph

        import openhands.agent_server.conversation_router as _conv_router

        _conv_router.get_default_tools = _get_default_tools_with_codegraph
    except Exception:
        pass  # best-effort secondary path; the defaults.py patch is primary


_install_default_tools_patch()
