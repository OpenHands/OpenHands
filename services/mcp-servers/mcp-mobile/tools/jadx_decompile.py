"""mobile_jadx_decompile."""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from shared.normalize import PathTraversalError, resolve_workspace_path
from shared.tool_result import err, ok
from tools._common import run_cmd, use_real_binaries

Runner = Callable[[Path, Path], Awaitable[tuple[int, str, str]]]


async def _default_runner(apk: Path, out_dir: Path) -> tuple[int, str, str]:
    return await run_cmd(
        "jadx",
        ["-d", str(out_dir), str(apk)],
        use_real=use_real_binaries(),
        timeout=300.0,
    )


async def run_jadx_decompile(
    *,
    engagement_id: str,
    apk_path: str,
    out_dir: str | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        resolved = resolve_workspace_path(apk_path)
        out = resolve_workspace_path(out_dir or f"{resolved.stem}_jadx")
    except PathTraversalError as exc:
        return err(exc.code, path=exc.path, message=str(exc))

    run = runner or _default_runner
    code, stdout, stderr = await run(resolved, out)
    if use_real_binaries() and code != 0:
        return err("jadx_failed", message=(stderr or stdout)[:300])
    return ok(
        {
            "tool": "mobile_jadx_decompile",
            "engagement_id": engagement_id,
            "apk_path": str(resolved),
            "out_dir": str(out),
            "output": (stdout or "decompiled").strip(),
        }
    )
