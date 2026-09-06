"""Codebase graph: nodes, four edge kinds, SQLite store, incremental indexer."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from parser_fallback import (
    EXTENSION_LANGUAGE,
    KIND_FILE,
    KIND_MODULE,
    SUPPORTED_LANGUAGES,
    detect_language,
    parse_source as fallback_parse_source,
)

GRAPH_DB_FILENAME = "codebase_graph.sqlite"
CONFIG_KEY = "config"
SOURCE_GRAPH = "graph"
EDGE_CALL = "call"
EDGE_IMPORT = "import"
EDGE_DEFINITION = "definition"
EDGE_REFERENCE = "reference"
EDGE_KINDS = (EDGE_CALL, EDGE_IMPORT, EDGE_DEFINITION, EDGE_REFERENCE)
QUERY_CALLERS = "callers"
QUERY_DEPS = "deps"
QUERY_USAGES = "usages"
QUERY_KINDS = (QUERY_CALLERS, QUERY_DEPS, QUERY_USAGES)
DEFAULT_QUERY_BUDGET = 50
SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    ".tmp",
    "coverage",
}
MAX_FILE_BYTES = 1_000_000
DEFAULT_GRAPH_BUDGET_LINES = 400
DEFAULT_MAX_CONTEXT_FILES = 12
DEFAULT_STALE_AFTER_MINUTES = 60
MODE_WARN = "warn"
MODE_STRICT = "strict"

ParseFn = Callable[[str, str, str], dict[str, Any]]

_active: "GraphStore | None" = None


class GraphError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status
        self.payload: dict[str, Any] = {}


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, GRAPH_DB_FILENAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "languages": list(SUPPORTED_LANGUAGES),
        "graph_budget_lines": DEFAULT_GRAPH_BUDGET_LINES,
        "max_context_files": DEFAULT_MAX_CONTEXT_FILES,
        "strict": False,
        "stale_after_minutes": DEFAULT_STALE_AFTER_MINUTES,
        "imported_project_path": None,
    }


def set_active_store(store: "GraphStore | None") -> None:
    global _active
    _active = store


def get_active_store() -> "GraphStore | None":
    return _active


def default_parse_file(path: str, source: str, language: str) -> dict[str, Any]:
    try:
        from tree_sitter_parser import parse_source as tree_sitter_parse

        return tree_sitter_parse(path, source, language)
    except Exception:
        return fallback_parse_source(path, source, language)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _node_id(kind: str, name: str, rel: str = "", line: int = 0, external: bool = False) -> str:
    if external:
        return f"external:{name}"
    return f"{kind}:{rel}:{name}:{line}"


class GraphStore:
    """SQLite-backed graph indexed from disk. Truth is the workspace tree."""

    def __init__(
        self,
        db_path: str = ":memory:",
        parse_file: ParseFn | None = None,
    ) -> None:
        self.db_path = db_path
        self.parse_file = parse_file or default_parse_file
        self._lock = threading.RLock()
        self._running = False
        self._status: dict[str, Any] = {
            "running": False,
            "files_indexed": 0,
            "symbols": 0,
            "edges": 0,
            "languages_used": [],
            "coverage": 0.0,
            "last_full_index_at": None,
            "last_error": None,
            "root": None,
        }
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                root TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT NOT NULL,
                mtime REAL NOT NULL,
                PRIMARY KEY (root, path)
            );
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                root TEXT NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                external INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                root TEXT NOT NULL,
                kind TEXT NOT NULL,
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                name TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_root_name ON nodes(root, name);
            CREATE INDEX IF NOT EXISTS idx_edges_root_kind ON edges(root, kind);
            """
        )
        self.conn.commit()
        if self._get_kv(CONFIG_KEY) is None:
            self._put_kv(CONFIG_KEY, default_config())

    def _get_kv(self, key: str) -> Any | None:
        row = self.conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def _put_kv(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_config(self) -> dict[str, Any]:
        config = default_config()
        stored = self._get_kv(CONFIG_KEY) or {}
        config.update(stored)
        return config

    def put_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self.get_config()
        allowed = set(default_config())
        for key, value in payload.items():
            if key not in allowed:
                raise GraphError(f"Unknown config key: {key}")
            config[key] = value
        languages = config.get("languages") or list(SUPPORTED_LANGUAGES)
        if not isinstance(languages, list) or any(not isinstance(item, str) for item in languages):
            raise GraphError("languages must be a list of strings")
        config["languages"] = languages
        config["enabled"] = bool(config.get("enabled"))
        config["strict"] = bool(config.get("strict"))
        config["graph_budget_lines"] = int(config.get("graph_budget_lines") or 0)
        config["max_context_files"] = int(config.get("max_context_files") or 0)
        config["stale_after_minutes"] = int(config.get("stale_after_minutes") or 0)
        self._put_kv(CONFIG_KEY, config)
        return config

    def import_project_config(self, path: str) -> dict[str, Any]:
        from project_config import load_project_config

        abs_path = os.path.abspath(path)
        config = self.get_config()
        if config.get("imported_project_path") == abs_path:
            return config
        data = load_project_config(abs_path)
        graph = (data.get("project") or {}).get("graph") or {}
        if not isinstance(graph, dict):
            graph = {}
        patch = {key: graph[key] for key in default_config() if key in graph}
        patch["imported_project_path"] = abs_path
        return self.put_config(patch)

    def status(self, root: str | None = None) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._status)
            target = os.path.abspath(root) if root else payload.get("root")
            if target:
                payload.update(self._counts(target))
                payload["root"] = target
            payload["running"] = self._running
            return payload

    def _counts(self, root: str) -> dict[str, Any]:
        files = self.conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE root = ?", (root,)
        ).fetchone()["n"]
        symbols = self.conn.execute(
            "SELECT COUNT(*) AS n FROM nodes WHERE root = ? AND kind NOT IN (?, ?)",
            (root, KIND_FILE, KIND_MODULE),
        ).fetchone()["n"]
        edges = self.conn.execute(
            "SELECT COUNT(*) AS n FROM edges WHERE root = ?", (root,)
        ).fetchone()["n"]
        langs = [
            row["language"]
            for row in self.conn.execute(
                "SELECT DISTINCT language FROM files WHERE root = ? ORDER BY language",
                (root,),
            )
        ]
        meta = self._get_kv(f"meta:{root}") or {}
        scanned = int(meta.get("files_scanned") or files)
        coverage = (files / scanned) if scanned else 0.0
        return {
            "files_indexed": files,
            "symbols": symbols,
            "edges": edges,
            "languages_used": langs,
            "coverage": coverage,
            "last_full_index_at": meta.get("last_full_index_at"),
            "last_error": meta.get("last_error"),
        }

    def list_nodes(self, root: str) -> list[dict[str, Any]]:
        root = os.path.abspath(root)
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM nodes WHERE root = ? ORDER BY id", (root,)
            ).fetchall()
        return [self._node_out(row) for row in rows]

    def list_edges(self, root: str) -> list[dict[str, Any]]:
        root = os.path.abspath(root)
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE root = ? ORDER BY id", (root,)
            ).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def _node_out(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data["external"] = bool(data.get("external"))
        return data

    def clear(self, root: str) -> dict[str, Any]:
        root = os.path.abspath(root)
        with self._lock:
            self.conn.execute("DELETE FROM edges WHERE root = ?", (root,))
            self.conn.execute("DELETE FROM nodes WHERE root = ?", (root,))
            self.conn.execute("DELETE FROM files WHERE root = ?", (root,))
            self.conn.commit()
            self._put_kv(
                f"meta:{root}",
                {
                    "last_full_index_at": None,
                    "last_error": None,
                    "files_scanned": 0,
                },
            )
        return self.status(root)

    def retrigger(self, root: str) -> dict[str, Any]:
        return self.index(root, full=True)

    def index(self, root: str, full: bool = False) -> dict[str, Any]:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise GraphError(f"root is not a directory: {root}")
        with self._lock:
            if self._running:
                return self.status(root)
            self._running = True
            self._status["running"] = True
            self._status["root"] = root
            self._status["last_error"] = None
        parsed = 0
        error: str | None = None
        try:
            config = self.get_config()
            allowed = set(config.get("languages") or SUPPORTED_LANGUAGES)
            scanned = list(self._scan(root, allowed))
            with self._lock:
                if full:
                    self.conn.execute("DELETE FROM edges WHERE root = ?", (root,))
                    self.conn.execute("DELETE FROM nodes WHERE root = ?", (root,))
                    self.conn.execute("DELETE FROM files WHERE root = ?", (root,))
                    self.conn.commit()
                    mtimes: dict[str, float] = {}
                else:
                    mtimes = {
                        row["path"]: row["mtime"]
                        for row in self.conn.execute(
                            "SELECT path, mtime FROM files WHERE root = ?", (root,)
                        )
                    }
                known = set(mtimes)
                seen: set[str] = set()
            for abs_path, rel, language, mtime in scanned:
                seen.add(rel)
                if not full and mtimes.get(rel) == mtime:
                    continue
                self._index_file(root, abs_path, rel, language, mtime)
                parsed += 1
            with self._lock:
                for stale in known - seen:
                    self._purge_file(root, stale)
                self._relink(root)
                meta = {
                    "files_scanned": len(scanned),
                    "last_error": None,
                    "last_full_index_at": utc_now()
                    if full
                    else (self._get_kv(f"meta:{root}") or {}).get("last_full_index_at"),
                }
                self._put_kv(f"meta:{root}", meta)
        except Exception as exc:
            error = str(exc)
            with self._lock:
                meta = self._get_kv(f"meta:{root}") or {}
                meta["last_error"] = error
                self._put_kv(f"meta:{root}", meta)
        finally:
            with self._lock:
                self._running = False
                self._status = {
                    **self.status(root),
                    "running": False,
                    "last_error": error,
                    "root": root,
                }
        payload = self.status(root)
        payload["files_parsed"] = parsed
        payload["last_error"] = error
        return payload

    def _scan(
        self, root: str, allowed: set[str]
    ) -> list[tuple[str, str, str, float]]:
        found: list[tuple[str, str, str, float]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")
            )
            for name in sorted(filenames):
                abs_path = os.path.join(dirpath, name)
                language = detect_language(abs_path)
                if language is None or language not in allowed:
                    continue
                try:
                    stat = os.stat(abs_path)
                except OSError:
                    continue
                if stat.st_size > MAX_FILE_BYTES:
                    continue
                rel = os.path.relpath(abs_path, root).replace("\\", "/")
                found.append((abs_path, rel, language, stat.st_mtime))
        return found

    def _node_ids_for_path(self, root: str, rel: str) -> list[str]:
        return [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM nodes WHERE root = ? AND path = ?", (root, rel)
            )
        ]

    def _purge_file(self, root: str, rel: str) -> None:
        ids = self._node_ids_for_path(root, rel)
        if ids:
            placeholders = ",".join("?" * len(ids))
            self.conn.execute(
                f"DELETE FROM edges WHERE root = ? AND (src_id IN ({placeholders}) OR dst_id IN ({placeholders}))",
                [root, *ids, *ids],
            )
        self.conn.execute("DELETE FROM nodes WHERE root = ? AND path = ?", (root, rel))
        self.conn.execute("DELETE FROM files WHERE root = ? AND path = ?", (root, rel))
        self.conn.commit()

    def _delete_file(self, root: str, rel: str) -> None:
        rows = self.conn.execute(
            "SELECT id, kind FROM nodes WHERE root = ? AND path = ?", (root, rel)
        ).fetchall()
        all_ids = [row["id"] for row in rows]
        symbol_ids = [
            row["id"]
            for row in rows
            if row["kind"] not in (KIND_FILE, KIND_MODULE)
        ]
        if all_ids:
            placeholders = ",".join("?" * len(all_ids))
            self.conn.execute(
                f"DELETE FROM edges WHERE root = ? AND src_id IN ({placeholders})",
                [root, *all_ids],
            )
        if symbol_ids:
            placeholders = ",".join("?" * len(symbol_ids))
            self.conn.execute(
                f"DELETE FROM nodes WHERE id IN ({placeholders})",
                symbol_ids,
            )
        self.conn.execute(
            "DELETE FROM files WHERE root = ? AND path = ?", (root, rel)
        )
        self.conn.commit()

    def _index_file(
        self,
        root: str,
        abs_path: str,
        rel: str,
        language: str,
        mtime: float,
    ) -> None:
        try:
            with open(abs_path, encoding="utf-8") as handle:
                source = handle.read()
            parsed = self.parse_file(abs_path, source, language)
        except Exception:
            parsed = fallback_parse_source(abs_path, source if "source" in locals() else "", language)
        with self._lock:
            self._delete_file(root, rel)
            self.conn.execute(
                "INSERT INTO files(root, path, language, mtime) VALUES (?, ?, ?, ?)",
                (root, rel, language, mtime),
            )
            file_id = _node_id(KIND_FILE, os.path.basename(rel), rel, 0)
            module_name = os.path.splitext(os.path.basename(rel))[0]
            module_id = _node_id(KIND_MODULE, module_name, rel, 0)
            self._upsert_node(
                file_id, root, KIND_FILE, os.path.basename(rel), rel, 1, 1, False
            )
            self._upsert_node(
                module_id, root, KIND_MODULE, module_name, rel, 1, 1, False
            )
            self._upsert_edge(
                root, EDGE_DEFINITION, file_id, module_id, module_name
            )
            symbol_ids: dict[str, str] = {}
            for item in parsed.get("symbols") or []:
                name = str(item.get("name") or "")
                if not name:
                    continue
                kind = str(item.get("kind") or "function")
                line = int(item.get("start_line") or 1)
                node_id = _node_id(kind, name, rel, line)
                symbol_ids[name] = node_id
                self._upsert_node(
                    node_id,
                    root,
                    kind,
                    name,
                    rel,
                    line,
                    int(item.get("end_line") or line),
                    False,
                )
                self._upsert_edge(root, EDGE_DEFINITION, file_id, node_id, name)
            for item in parsed.get("imports") or []:
                spec = str(item.get("name") or "")
                if not spec:
                    continue
                dest_id, dest_name, external = self._resolve_import(root, rel, spec)
                if external:
                    self._upsert_node(
                        dest_id, root, KIND_MODULE, dest_name, None, 1, 1, True
                    )
                self._upsert_edge(root, EDGE_IMPORT, file_id, dest_id, dest_name)
            for item in parsed.get("calls") or []:
                name = str(item.get("name") or "")
                if not name:
                    continue
                src_id = symbol_ids.get(self._enclosing_symbol(parsed, item)) or file_id
                dest = self._find_symbol(root, name) or _node_id(
                    KIND_MODULE, name, "", 0, external=True
                )
                if dest.startswith("external:"):
                    self._upsert_node(dest, root, KIND_MODULE, name, None, 1, 1, True)
                self._upsert_edge(root, EDGE_CALL, src_id, dest, name)
            for item in parsed.get("references") or []:
                name = str(item.get("name") or "")
                if not name:
                    continue
                dest = self._find_symbol(root, name) or _node_id(
                    KIND_MODULE, name, "", 0, external=True
                )
                if dest.startswith("external:"):
                    self._upsert_node(dest, root, KIND_MODULE, name, None, 1, 1, True)
                self._upsert_edge(root, EDGE_REFERENCE, file_id, dest, name)
            self.conn.commit()

    def _enclosing_symbol(self, parsed: dict[str, Any], item: dict[str, Any]) -> str:
        line = int(item.get("start_line") or 0)
        enclosing = ""
        best = -1
        for symbol in parsed.get("symbols") or []:
            start = int(symbol.get("start_line") or 0)
            if start <= line and start >= best:
                enclosing = str(symbol.get("name") or "")
                best = start
        return enclosing

    def _find_symbol(self, root: str, name: str) -> str | None:
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE root = ? AND name = ? "
            "AND kind NOT IN (?, ?) ORDER BY start_line LIMIT 1",
            (root, name, KIND_FILE, KIND_MODULE),
        ).fetchone()
        if row:
            return row["id"]
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE root = ? AND name = ? LIMIT 1",
            (root, name),
        ).fetchone()
        return row["id"] if row else None

    def _resolve_import(self, root: str, from_rel: str, spec: str) -> tuple[str, str, bool]:
        name = spec.split("/")[-1].split(".")[0] or spec
        if spec.startswith("http://") or spec.startswith("https://"):
            node_id = _node_id(KIND_MODULE, spec, "", 0, external=True)
            return node_id, spec, True
        candidates = self._import_candidates(root, from_rel, spec)
        for abs_path in candidates:
            if os.path.isfile(abs_path):
                rel = os.path.relpath(abs_path, root).replace("\\", "/")
                module_name = os.path.splitext(os.path.basename(rel))[0]
                module_id = _node_id(KIND_MODULE, module_name, rel, 0)
                existing = self.conn.execute(
                    "SELECT id FROM nodes WHERE id = ?", (module_id,)
                ).fetchone()
                if existing is None:
                    self._upsert_node(
                        module_id,
                        root,
                        KIND_MODULE,
                        module_name,
                        rel,
                        1,
                        1,
                        False,
                    )
                return module_id, module_name, False
        node_id = _node_id(KIND_MODULE, name, "", 0, external=True)
        return node_id, name, True

    def _import_candidates(self, root: str, from_rel: str, spec: str) -> list[str]:
        bases: list[str] = []
        dirname = os.path.dirname(os.path.join(root, from_rel))
        if spec.startswith("."):
            bases.append(os.path.normpath(os.path.join(dirname, spec)))
        else:
            bases.append(os.path.normpath(os.path.join(dirname, spec)))
            bases.append(os.path.normpath(os.path.join(root, spec.replace(".", os.sep))))
            bases.append(os.path.normpath(os.path.join(root, spec)))
        found: list[str] = []
        extras = list(EXTENSION_LANGUAGE)
        for base in bases:
            found.append(base)
            for ext in extras:
                found.append(base + ext)
            for index_name in ("index.js", "index.ts", "index.tsx", "__init__.py"):
                found.append(os.path.join(base, index_name))
        return found

    def _relink(self, root: str) -> None:
        rows = self.conn.execute(
            "SELECT id, name, dst_id FROM edges WHERE root = ? AND kind IN (?, ?)",
            (root, EDGE_CALL, EDGE_REFERENCE),
        ).fetchall()
        for row in rows:
            found = self._find_symbol(root, row["name"])
            if found and found != row["dst_id"]:
                self.conn.execute(
                    "UPDATE edges SET dst_id = ? WHERE id = ?",
                    (found, row["id"]),
                )
        self.conn.commit()

    def _upsert_node(
        self,
        node_id: str,
        root: str,
        kind: str,
        name: str,
        path: str | None,
        start_line: int,
        end_line: int,
        external: bool,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO nodes(id, root, kind, name, qualified_name, path, start_line, end_line, external)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                name = excluded.name,
                path = excluded.path,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                external = excluded.external
            """,
            (
                node_id,
                root,
                kind,
                name,
                name,
                path,
                start_line,
                end_line,
                1 if external else 0,
            ),
        )

    def _upsert_edge(
        self, root: str, kind: str, src_id: str, dst_id: str, name: str
    ) -> None:
        edge_id = f"{kind}:{src_id}->{dst_id}:{name}"
        self.conn.execute(
            """
            INSERT INTO edges(id, root, kind, src_id, dst_id, name)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (edge_id, root, kind, src_id, dst_id, name),
        )

    def _query_envelope(
        self,
        query: str,
        root: str,
        symbol: str | None,
        result: list[dict[str, Any]],
        file: str | None = None,
    ) -> dict[str, Any]:
        status = self.status(root)
        stale = False
        last = status.get("last_full_index_at")
        config = self.get_config()
        if last:
            try:
                indexed = datetime.fromisoformat(last)
                if indexed.tzinfo is None:
                    indexed = indexed.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - indexed).total_seconds() / 60
                stale = age > int(config.get("stale_after_minutes") or DEFAULT_STALE_AFTER_MINUTES)
            except ValueError:
                stale = False
        return {
            "query": query,
            "symbol": symbol,
            "file": file,
            "result": result,
            "source": SOURCE_GRAPH,
            "status": "stale" if stale else ("empty" if not last else "ok"),
            "last_full_index_at": last,
            "coverage": status.get("coverage") or 0.0,
        }

    def query(
        self,
        kind: str,
        symbol: str | None = None,
        file: str | None = None,
        root: str | None = None,
        budget: int = DEFAULT_QUERY_BUDGET,
    ) -> dict[str, Any]:
        if kind not in QUERY_KINDS:
            raise GraphError(f"Unknown query: {kind}")
        if not root:
            raise GraphError("root is required")
        root = os.path.abspath(root)
        limit = max(1, int(budget or DEFAULT_QUERY_BUDGET))
        if kind == QUERY_CALLERS:
            if not symbol:
                raise GraphError("symbol is required")
            rows = self.conn.execute(
                """
                SELECT n.* FROM edges e
                JOIN nodes n ON n.id = e.src_id
                WHERE e.root = ? AND e.kind = ? AND e.name = ?
                ORDER BY n.id LIMIT ?
                """,
                (root, EDGE_CALL, symbol, limit),
            ).fetchall()
            result = [self._node_out(row) for row in rows]
            return self._query_envelope(kind, root, symbol, result)
        if kind == QUERY_DEPS:
            if not file:
                raise GraphError("file is required")
            rel = file.replace("\\", "/")
            rows = self.conn.execute(
                """
                SELECT n.* FROM edges e
                JOIN nodes src ON src.id = e.src_id
                JOIN nodes n ON n.id = e.dst_id
                WHERE e.root = ? AND e.kind = ? AND src.path = ?
                ORDER BY n.id LIMIT ?
                """,
                (root, EDGE_IMPORT, rel, limit),
            ).fetchall()
            result = [self._node_out(row) for row in rows]
            return self._query_envelope(kind, root, symbol, result, file=rel)
        rows = self.conn.execute(
            """
            SELECT n.*, e.kind AS edge_kind FROM edges e
            JOIN nodes n ON n.id = e.src_id
            WHERE e.root = ? AND e.kind IN (?, ?) AND e.name = ?
            ORDER BY n.id LIMIT ?
            """,
            (root, EDGE_REFERENCE, EDGE_CALL, symbol or "", limit),
        ).fetchall()
        result = []
        for row in rows:
            item = self._node_out(row)
            item["kind"] = row["edge_kind"]
            result.append(item)
        return self._query_envelope(kind, root, symbol, result)

    def definitions(self, symbol: str, root: str, budget: int = DEFAULT_QUERY_BUDGET) -> dict[str, Any]:
        if not symbol:
            raise GraphError("symbol is required")
        root = os.path.abspath(root)
        limit = max(1, int(budget or DEFAULT_QUERY_BUDGET))
        rows = self.conn.execute(
            """
            SELECT * FROM nodes
            WHERE root = ? AND name LIKE ? AND kind NOT IN (?, ?)
            ORDER BY id LIMIT ?
            """,
            (root, f"%{symbol}%", KIND_FILE, KIND_MODULE, limit),
        ).fetchall()
        return self._query_envelope(
            "definitions", root, symbol, [self._node_out(row) for row in rows]
        )
