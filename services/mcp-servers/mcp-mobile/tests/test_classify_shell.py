"""AppSec HIGH-1 — classify_shell_command metachar / boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.adb_shell import classify_shell_command, run_adb_shell


# @spec PROJETOSIN-190 — AppSec HIGH-1 metachar / prefix boundary
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pm list packages", "safe"),
        ("pm path com.example.app", "safe"),
        ("logcat -d", "safe"),
        ("dumpsys package com.example", "safe"),
        ("getprop ro.build.version.release", "safe"),
        ("settings get secure android_id", "safe"),
        ("id", "safe"),
        ("ls /sdcard", "safe"),
        ("am start -n com.example/.Main", "safe"),
        ("pm uninstall com.example.app", "mutant"),
        ("settings put secure foo bar", "mutant"),
        ("rm -rf /", "blocked"),
        # PoCs: chaining must never be safe
        ("ls /; pm uninstall com.victim", "denied"),
        ("id; pm uninstall com.victim", "denied"),
        ("ls /data && pm uninstall com.x", "denied"),
        ("am start -n x/.Y; pm uninstall com.x", "denied"),
        ("dumpsys; settings put secure foo bar", "denied"),
        ("pm list packages | grep victim", "denied"),
        ("id`pm uninstall com.victim`", "denied"),
        ("id$(pm uninstall com.victim)", "denied"),
        # Accidental prefix: allowlist has "id", not "identical"
        ("identical", "denied"),
        ("lsfoo", "denied"),
        ("whoamiEvil", "denied"),
    ],
)
def test_classify_shell_command_boundary_and_metachar(command: str, expected: str):
    assert classify_shell_command(command) == expected


@pytest.mark.asyncio
async def test_chained_mutant_does_not_bypass_gate():
    """Chained uninstall must be denied — never silent safe execution."""
    body = json.loads(
        await run_adb_shell(
            engagement_id="00000000-0000-4000-8000-000000000190",
            command="ls /; pm uninstall com.victim",
        )
    )
    assert body["ok"] is False
    assert body["error"] == "adb_shell_not_allowlisted"


@pytest.mark.asyncio
async def test_identical_not_treated_as_id():
    body = json.loads(
        await run_adb_shell(
            engagement_id="00000000-0000-4000-8000-000000000190",
            command="identical",
        )
    )
    assert body["ok"] is False
    assert body["error"] == "adb_shell_not_allowlisted"


@pytest.mark.asyncio
async def test_frida_script_outside_workspace_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PENTEST_WORKSPACE_DIR", str(tmp_path))
    outside = tmp_path.parent / "evil.js"
    outside.write_text("Java.perform(function(){});", encoding="utf-8")

    from tools.frida_attach import run_frida_attach

    body = json.loads(
        await run_frida_attach(
            engagement_id="00000000-0000-4000-8000-000000000190",
            package="com.example.app",
            script=str(outside),
        )
    )
    assert body["ok"] is False
    assert body["error"] == "path_traversal"
