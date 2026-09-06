"""Provider → CLI adapter dispatch.

Mirrors ``cursor_tool.py``: subprocess + PATH fallbacks + ``--output-format json``
and ``--model <resolved.model>``. Missing binaries are unusable so the router
fallback chain can skip them. Does not reimplement the Cursor tool — it calls
the same binary shape, and prefers ``cursor_tool.CursorExecutor`` when the SDK
is importable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Callable

PROVIDER_CURSOR = "cursor-cli"
PROVIDER_CURSOR_ALIAS = "cursor"
PROVIDER_CLAUDE_CODE = "claude-code"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENCODE = "opencode"
PROVIDER_OLLAMA = "ollama"

RunFn = Callable[[list[str], str | None, float], dict[str, Any]]


class AdapterError(RuntimeError):
    def __init__(self, message: str, *, unusable: bool = False) -> None:
        super().__init__(message)
        self.unusable = unusable


def _which(names: tuple[str, ...], fallbacks: tuple[str, ...]) -> str | None:
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    for candidate in fallbacks:
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded
    return None


def find_cursor_binary() -> str | None:
    try:
        from cursor_tool import CursorExecutor

        return CursorExecutor._find_agent()
    except Exception:
        return _which(
            ("agent", "cursor-agent"),
            ("~/.local/bin/agent", "/usr/local/bin/agent", "/opt/homebrew/bin/agent"),
        )


def find_claude_binary() -> str | None:
    return _which(
        ("claude",),
        ("~/.local/bin/claude", "/usr/local/bin/claude", "/opt/homebrew/bin/claude"),
    )


def find_opencode_binary() -> str | None:
    try:
        from opencode_tool import OpenCodeExecutor

        finder = getattr(OpenCodeExecutor, "_find_opencode", None)
        if finder:
            return finder()
    except Exception:
        pass
    return _which(
        ("opencode",),
        (
            "~/.opencode/bin/opencode",
            "/usr/local/bin/opencode",
            "/opt/homebrew/bin/opencode",
        ),
    )


def find_ollama_binary() -> str | None:
    return _which(
        ("ollama",),
        ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama"),
    )


def default_run(cmd: list[str], cwd: str | None, timeout_s: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(f"timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise AdapterError(f"failed to spawn: {exc}", unusable=True) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        raise AdapterError(f"exited {proc.returncode}: {tail}")
    return {
        "ok": True,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "cmd": cmd,
    }


class ProviderAdapter:
    provider_key: str

    def find_binary(self) -> str | None:
        raise NotImplementedError

    def is_usable(self) -> bool:
        return self.find_binary() is not None

    def build_cmd(self, binary: str, prompt: str, model: str | None) -> list[str]:
        raise NotImplementedError

    def run(
        self,
        prompt: str,
        model: str | None = None,
        *,
        cwd: str | None = None,
        timeout_s: float = 900,
        runner: RunFn | None = None,
    ) -> dict[str, Any]:
        binary = self.find_binary()
        if binary is None:
            raise AdapterError(f"{self.provider_key} binary is not installed", unusable=True)
        cmd = self.build_cmd(binary, prompt, model)
        run = runner or default_run
        result = run(cmd, cwd, timeout_s)
        result["provider_key"] = self.provider_key
        result["model"] = model
        return result


class CursorAdapter(ProviderAdapter):
    provider_key = PROVIDER_CURSOR

    def find_binary(self) -> str | None:
        return find_cursor_binary()

    def build_cmd(self, binary: str, prompt: str, model: str | None) -> list[str]:
        cmd = [binary, "-p", prompt, "--output-format", "json", "--force"]
        if model:
            cmd += ["--model", model]
        return cmd


class ClaudeCodeAdapter(ProviderAdapter):
    provider_key = PROVIDER_CLAUDE_CODE

    def find_binary(self) -> str | None:
        return find_claude_binary()

    def build_cmd(self, binary: str, prompt: str, model: str | None) -> list[str]:
        cmd = [binary, "-p", prompt, "--output-format", "json"]
        if model:
            cmd += ["--model", model]
        return cmd


class OpenCodeAdapter(ProviderAdapter):
    provider_key = PROVIDER_OPENCODE

    def find_binary(self) -> str | None:
        return find_opencode_binary()

    def build_cmd(self, binary: str, prompt: str, model: str | None) -> list[str]:
        cmd = [binary, "run", "--format", "json", prompt]
        if model:
            cmd += ["--model", model]
        return cmd


class OllamaAdapter(ProviderAdapter):
    provider_key = PROVIDER_OLLAMA

    def find_binary(self) -> str | None:
        return find_ollama_binary()

    def build_cmd(self, binary: str, prompt: str, model: str | None) -> list[str]:
        tag = model.split("/", 1)[-1] if model else "llama3.2"
        return [binary, "run", tag, prompt]


ADAPTERS: dict[str, ProviderAdapter] = {
    PROVIDER_CURSOR: CursorAdapter(),
    PROVIDER_CURSOR_ALIAS: CursorAdapter(),
    PROVIDER_CLAUDE_CODE: ClaudeCodeAdapter(),
    PROVIDER_ANTHROPIC: ClaudeCodeAdapter(),
    PROVIDER_OPENCODE: OpenCodeAdapter(),
    PROVIDER_OLLAMA: OllamaAdapter(),
}


def adapter_for(provider_key: str) -> ProviderAdapter | None:
    return ADAPTERS.get(provider_key)


def usable_providers(
    *,
    connected: list[str] | None = None,
    finder: Callable[[str], bool] | None = None,
) -> list[str]:
    keys = list(dict.fromkeys(ADAPTERS))
    if connected is not None:
        keys = [key for key in keys if key in connected]
    result = []
    for key in keys:
        adapter = ADAPTERS[key]
        ok = finder(key) if finder else adapter.is_usable()
        if ok:
            result.append(key)
    return result


def dispatch(
    provider_key: str,
    prompt: str,
    model: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    adapter = adapter_for(provider_key)
    if adapter is None:
        raise AdapterError(f"no adapter for {provider_key}", unusable=True)
    return adapter.run(prompt, model, **kwargs)
