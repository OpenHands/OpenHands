from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import get_settings
from app.models.engagement import Engagement, ScopeRule

ComposeRunner = Callable[[list[str], Path], Awaitable[int]]

DRY_RUN_MOBSF_API_KEY = "test-mobsf-key"


async def _default_runner(args: list[str], cwd: Path) -> int:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return int(proc.returncode or 0)


RUNTIME_TEMPLATES = {
    "web": "compose-web-runtime.yml.j2",
    "network": "compose-network-runtime.yml.j2",
    "mobile": "compose-mobile-runtime.yml.j2",
    "sast": "compose-sast-runtime.yml.j2",
}


def build_mobile_network_metadata(project: str) -> dict:
    """Internal endpoints for UI proxy (PROJETOSIN-192) — never host-published."""
    return {
        "emulator": {
            "adb": f"{project}-emulator:5555",
            "vnc_internal": f"http://{project}-emulator:6080",
            "service_name": f"{project}-emulator",
        },
        "mobsf": {
            "url_internal": f"http://{project}-mobsf:8000",
        },
    }


def kvm_device_available() -> bool:
    return Path("/dev/kvm").exists()


class RuntimeProvisioner:
    def __init__(
        self,
        *,
        runner: ComposeRunner | None = None,
        templates_dir: Path | None = None,
        dry_run: bool | None = None,
    ):
        settings = get_settings()
        self.dry_run = settings.provisioner_dry_run if dry_run is None else dry_run
        self.runner = runner or _default_runner
        self.work_root = Path(settings.compose_work_dir)
        self.android_emulator_image = settings.android_emulator_image
        self.mobsf_image = settings.mobsf_image
        self.allow_slow_emulator = settings.allow_slow_emulator
        self.mcp_mobile_cmd = settings.mcp_mobile_cmd
        base = templates_dir or (
            Path(__file__).resolve().parents[1] / "templates"
        )
        self.env = Environment(
            loader=FileSystemLoader(str(base)),
            autoescape=select_autoescape(enabled_extensions=()),
        )
        self.last_commands: list[list[str]] = []
        self.last_metadata: dict | None = None

    def project_name(self, engagement: Engagement) -> str:
        return f"eng-{str(engagement.id).replace('-', '')[:8]}"

    def _mobsf_api_key(self) -> str:
        if self.dry_run:
            return DRY_RUN_MOBSF_API_KEY
        return secrets.token_urlsafe(32)

    def _resolve_kvm(self) -> bool:
        if self.dry_run:
            # Dry-run assumes KVM path for stable fixtures; host check is live-only.
            return True
        available = kvm_device_available()
        if available:
            return True
        if self.allow_slow_emulator:
            return False
        raise RuntimeError(
            "Android emulator requires /dev/kvm on the Docker host. "
            "Set ALLOW_SLOW_EMULATOR=1 for software fallback (very slow), "
            "or provision on a Linux host with KVM. "
            "Windows/macOS Docker Desktop: see docker/runtimes/README.md."
        )

    def _render(
        self, engagement: Engagement, scope_rules: list[ScopeRule]
    ) -> str:
        template_name = RUNTIME_TEMPLATES[engagement.runtime_profile]
        short = self.project_name(engagement)
        allow = [
            {"type": r.target_type, "value": r.target_value}
            for r in scope_rules
            if r.rule_type == "allow"
        ]
        deny = [
            {"type": r.target_type, "value": r.target_value}
            for r in scope_rules
            if r.rule_type == "deny"
        ]
        ctx: dict = {
            "project": short,
            "network_internal": f"{short}-internal",
            "network_egress": f"{short}-egress",
            "volume_prefix": short,
            "allow_rules": allow,
            "deny_rules": deny,
            "autonomy_mode": engagement.autonomy_mode or "semi_autonomous",
            "runtime_image": (
                f"ghcr.io/heimdall/runtime-{engagement.runtime_profile}:latest"
            ),
        }
        if engagement.runtime_profile == "mobile":
            ctx.update(
                {
                    "android_emulator_image": self.android_emulator_image,
                    "mobsf_image": self.mobsf_image,
                    "mobsf_api_key": self._mobsf_api_key(),
                    "mcp_mobile_cmd": self.mcp_mobile_cmd,
                    "kvm_available": self._resolve_kvm(),
                    "emulator_device": "Samsung Galaxy S10",
                }
            )
        return self.env.get_template(template_name).render(**ctx)

    async def provision(
        self, engagement: Engagement, scope_rules: list[ScopeRule]
    ) -> str:
        project = self.project_name(engagement)
        work = self.work_root / project
        work.mkdir(parents=True, exist_ok=True)
        compose_path = work / "docker-compose.yml"
        compose_path.write_text(
            self._render(engagement, scope_rules), encoding="utf-8"
        )
        if engagement.runtime_profile == "mobile":
            meta = build_mobile_network_metadata(project)
            self.last_metadata = meta
            (work / "runtime-metadata.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
        else:
            self.last_metadata = None
        args = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(compose_path),
            "up",
            "-d",
        ]
        self.last_commands.append(args)
        if not self.dry_run:
            code = await self.runner(args, work)
            if code != 0:
                raise RuntimeError(f"docker compose up failed with {code}")
        return project

    async def teardown(self, engagement: Engagement) -> None:
        project = engagement.sandbox_compose_project or self.project_name(engagement)
        work = self.work_root / project
        compose_path = work / "docker-compose.yml"
        args = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(compose_path),
            "down",
            "-v",
        ]
        self.last_commands.append(args)
        if not self.dry_run and compose_path.exists():
            code = await self.runner(args, work)
            if code != 0:
                raise RuntimeError(f"docker compose down failed with {code}")

    async def rewrite_compose(
        self, engagement: Engagement, scope_rules: list[ScopeRule]
    ) -> Path | None:
        """Rewrite compose YAML with current engagement fields (e.g. autonomy).

        Does not restart containers — callers decide applied vs pending_restart.
        """
        project = engagement.sandbox_compose_project or self.project_name(engagement)
        work = self.work_root / project
        if not work.exists() and not self.dry_run:
            return None
        work.mkdir(parents=True, exist_ok=True)
        compose_path = work / "docker-compose.yml"
        compose_path.write_text(
            self._render(engagement, scope_rules), encoding="utf-8"
        )
        return compose_path
