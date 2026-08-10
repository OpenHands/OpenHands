from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import get_settings
from app.models.engagement import Engagement, ScopeRule

ComposeRunner = Callable[[list[str], Path], Awaitable[int]]


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
        base = templates_dir or (
            Path(__file__).resolve().parents[1] / "templates"
        )
        self.env = Environment(
            loader=FileSystemLoader(str(base)),
            autoescape=select_autoescape(enabled_extensions=()),
        )
        self.last_commands: list[list[str]] = []

    def project_name(self, engagement: Engagement) -> str:
        return f"eng-{str(engagement.id).replace('-', '')[:8]}"

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
        return self.env.get_template(template_name).render(
            project=short,
            network_internal=f"{short}-internal",
            network_egress=f"{short}-egress",
            volume_prefix=short,
            allow_rules=allow,
            deny_rules=deny,
            runtime_image=f"ghcr.io/heimdall/runtime-{engagement.runtime_profile}:latest",
        )

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
        args = ["docker", "compose", "-p", project, "-f", str(compose_path), "up", "-d"]
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
