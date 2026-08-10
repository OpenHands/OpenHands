"""mcp-mobile — stdio MCP server for MobSF + ADB/Frida (PROJETOSIN-190).

Requires capability ``pentest.mobile.dynamic`` for session registration (launcher).
Autonomy comes from ``PENTEST_AUTONOMY_MODE`` (server-side), never tool args.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mcp.server.fastmcp import FastMCP

from tools.adb_connect import run_adb_connect, run_adb_devices
from tools.adb_install import run_adb_install
from tools.adb_shell import run_adb_shell
from tools.apktool_decode import run_apktool_decode
from tools.frida_attach import run_frida_attach, run_frida_list
from tools.jadx_decompile import run_jadx_decompile
from tools.mobsf_dynamic import run_mobsf_dynamic
from tools.mobsf_static import run_mobsf_static

# Capability required for agent-server / launcher registration.
REQUIRED_CAPABILITY = "pentest.mobile.dynamic"

mcp = FastMCP("mcp-mobile")


@mcp.tool()
async def mobile_mobsf_static(engagement_id: str, apk_path: str) -> str:
    """Upload APK to MobSF, run static scan, post findings."""
    return await run_mobsf_static(engagement_id=engagement_id, apk_path=apk_path)


@mcp.tool()
async def mobile_mobsf_dynamic(
    engagement_id: str,
    apk_path: str | None = None,
    package: str | None = None,
    confirmation_token: str | None = None,
) -> str:
    """Run MobSF dynamic analysis (confirmation in semi_autonomous).

    Autonomy comes from PENTEST_AUTONOMY_MODE (server-side), not tool args.
    """
    return await run_mobsf_dynamic(
        engagement_id=engagement_id,
        apk_path=apk_path,
        package=package,
        confirmation_token=confirmation_token,
    )


@mcp.tool()
async def mobile_adb_connect(
    host: str | None = None,
    port: str | None = None,
) -> str:
    """adb connect to ADB_HOST:ADB_PORT (or overrides)."""
    return await run_adb_connect(host=host, port=port)


@mcp.tool()
async def mobile_adb_devices() -> str:
    """List ADB devices."""
    return await run_adb_devices()


@mcp.tool()
async def mobile_adb_install(
    engagement_id: str,
    apk_path: str,
    confirmation_token: str | None = None,
) -> str:
    """Install APK via adb (confirmation in semi_autonomous).

    Autonomy comes from PENTEST_AUTONOMY_MODE (server-side), not tool args.
    """
    return await run_adb_install(
        engagement_id=engagement_id,
        apk_path=apk_path,
        confirmation_token=confirmation_token,
    )


@mcp.tool()
async def mobile_adb_shell(
    engagement_id: str,
    command: str,
    confirmation_token: str | None = None,
) -> str:
    """Run allowlisted adb shell; mutant commands require confirmation.

    Autonomy comes from PENTEST_AUTONOMY_MODE (server-side), not tool args.
    """
    return await run_adb_shell(
        engagement_id=engagement_id,
        command=command,
        confirmation_token=confirmation_token,
    )


@mcp.tool()
async def mobile_frida_list() -> str:
    """List processes via frida-ps -U."""
    return await run_frida_list()


@mcp.tool()
async def mobile_frida_attach(
    engagement_id: str,
    package: str,
    script: str | None = None,
    confirmation_token: str | None = None,
) -> str:
    """Attach Frida to a package (confirmation in semi_autonomous).

    Autonomy comes from PENTEST_AUTONOMY_MODE (server-side), not tool args.
    """
    return await run_frida_attach(
        engagement_id=engagement_id,
        package=package,
        script=script,
        confirmation_token=confirmation_token,
    )


@mcp.tool()
async def mobile_apktool_decode(
    engagement_id: str,
    apk_path: str,
    out_dir: str | None = None,
) -> str:
    """Decode APK with apktool under the engagement workspace."""
    return await run_apktool_decode(
        engagement_id=engagement_id, apk_path=apk_path, out_dir=out_dir
    )


@mcp.tool()
async def mobile_jadx_decompile(
    engagement_id: str,
    apk_path: str,
    out_dir: str | None = None,
) -> str:
    """Decompile APK with jadx under the engagement workspace."""
    return await run_jadx_decompile(
        engagement_id=engagement_id, apk_path=apk_path, out_dir=out_dir
    )


def list_tool_names() -> list[str]:
    """Test helper — FastMCP registers tools as module-level functions."""
    return [
        "mobile_mobsf_static",
        "mobile_mobsf_dynamic",
        "mobile_adb_connect",
        "mobile_adb_devices",
        "mobile_adb_install",
        "mobile_adb_shell",
        "mobile_frida_list",
        "mobile_frida_attach",
        "mobile_apktool_decode",
        "mobile_jadx_decompile",
    ]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
