#!/usr/bin/env python3
"""Mirror of .github/workflows Check Version Consistency job. Run from repo root: python scripts/check-version-consistency.py"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import tomllib

    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. pyproject.toml ───────────────────────────────────────────────────
    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    version = pyproject["tool"]["poetry"]["version"]
    major_minor = ".".join(version.split(".")[:2])
    print(f"pyproject.toml version: {version} (major.minor: {major_minor})")

    # ── 2. frontend/package.json ───────────────────────────────────────────
    with (ROOT / "frontend" / "package.json").open(encoding="utf-8") as f:
        pkg = json.load(f)
    if pkg["version"] != version:
        errors.append(
            f"frontend/package.json version is '{pkg['version']}', expected '{version}'"
        )
    else:
        print(f"  OK frontend/package.json: {pkg['version']}")

    # ── 3. frontend/package-lock.json ──────────────────────────────────────
    with (ROOT / "frontend" / "package-lock.json").open(encoding="utf-8") as f:
        lock = json.load(f)
    for key, val in [
        ("root.version", lock.get("version")),
        ('packages[""].version', lock.get("packages", {}).get("", {}).get("version")),
    ]:
        if val != version:
            errors.append(
                f"frontend/package-lock.json {key} is '{val}', expected '{version}'"
            )
        else:
            print(f"  OK frontend/package-lock.json {key}: {val}")

    # ── 4. Compose + sandbox_spec_service agent-server tag ─────────────────
    repo_pattern = re.compile(r"AGENT_SERVER_IMAGE_REPOSITORY[^}]*:-([^}]+)")
    tag_pattern = re.compile(r"AGENT_SERVER_IMAGE_TAG:-([^}]+)")

    sandbox_path = ROOT / "openhands" / "app_server" / "sandbox" / "sandbox_spec_service.py"
    sandbox_src = sandbox_path.read_text(encoding="utf-8")
    canon_m = re.search(
        r"AGENT_SERVER_IMAGE\s*=\s*['\"]ghcr\.io/openhands/agent-server:([^'\"]+)['\"]",
        sandbox_src,
    )
    if not canon_m:
        errors.append(
            "openhands/app_server/sandbox/sandbox_spec_service.py: "
            "cannot parse AGENT_SERVER_IMAGE tag"
        )
        expected_tag = None
    else:
        expected_tag = canon_m.group(1)
        print(
            f"  OK canonical agent-server tag (sandbox_spec_service.py): {expected_tag}"
        )

    for filepath in ["docker-compose.yml", "containers/dev/compose.yml"]:
        path = ROOT / filepath
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            warnings.append(f"{filepath}: file not found")
            continue

        repos = repo_pattern.findall(content)
        tags = tag_pattern.findall(content)

        if not repos:
            warnings.append(f"{filepath}: no AGENT_SERVER_IMAGE_REPOSITORY default found")
        else:
            repo = repos[0]
            if "agent-server" not in repo:
                errors.append(
                    f"{filepath}: AGENT_SERVER_IMAGE_REPOSITORY defaults to '{repo}', "
                    f"expected an agent-server image (not runtime)"
                )
            else:
                print(f"  OK {filepath} image repository: {repo}")

        if not tags:
            warnings.append(f"{filepath}: no AGENT_SERVER_IMAGE_TAG default found")
        else:
            tag = tags[0]
            if not tag:
                errors.append(f"{filepath}: AGENT_SERVER_IMAGE_TAG default is empty")
            elif expected_tag is not None and tag != expected_tag:
                errors.append(
                    f"{filepath}: AGENT_SERVER_IMAGE_TAG default is '{tag}', "
                    f"expected '{expected_tag}' (match sandbox_spec_service.AGENT_SERVER_IMAGE)"
                )
            else:
                print(f"  OK {filepath} image tag: {tag}")

    print()
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
        print()

    if errors:
        print("FAILED: Version inconsistencies found:\n")
        for e in errors:
            print(f"  - {e}")
        print(
            "\nAll version numbers and Docker image tags must be consistent."
            "\nSee .agents/skills/update-sdk/SKILL.md for the full checklist."
        )
        return 1

    print("OK: All version numbers and Docker image tags are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
