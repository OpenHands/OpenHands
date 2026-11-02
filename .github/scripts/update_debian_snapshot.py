#!/usr/bin/env python3
"""Advance the agent-server Debian snapshot after a seven-day observation period."""

from __future__ import annotations

import argparse
import re
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path


MINIMUM_AGE = timedelta(days=7)
SNAPSHOT_RE = re.compile(r"(?m)^ARG DEBIAN_SNAPSHOT=(\d{8}T\d{6}Z)$")
ARCHIVES = ("debian", "debian-security")


def eligible_snapshot(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cutoff = now.astimezone(UTC) - MINIMUM_AGE
    return cutoff.replace(hour=0, minute=0, second=0, microsecond=0)


def format_snapshot(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def validate_snapshot_age(snapshot: datetime, now: datetime) -> None:
    age = now.astimezone(UTC) - snapshot.astimezone(UTC)
    if age < MINIMUM_AGE:
        raise ValueError(f"snapshot is only {age} old; minimum age is {MINIMUM_AGE}")


def verify_snapshot(snapshot: str) -> None:
    for archive in ARCHIVES:
        url = f"https://snapshot.debian.org/archive/{archive}/{snapshot}/"
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(f"{url} returned HTTP {response.status}")
        except OSError as exc:
            raise ValueError(f"unable to verify {url}: {exc}") from exc


def update_dockerfile(path: Path, snapshot: str) -> bool:
    text = path.read_text(encoding="utf-8")
    matches = SNAPSHOT_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one DEBIAN_SNAPSHOT in {path}")
    updated = SNAPSHOT_RE.sub(f"ARG DEBIAN_SNAPSHOT={snapshot}", text)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument(
        "--now",
        type=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
        default=datetime.now(UTC),
        help="UTC reference time for deterministic testing",
    )
    parser.add_argument("--skip-network-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_time = eligible_snapshot(args.now)
    validate_snapshot_age(snapshot_time, args.now)
    snapshot = format_snapshot(snapshot_time)
    if not args.skip_network_check:
        verify_snapshot(snapshot)
    changed = update_dockerfile(args.dockerfile, snapshot)
    print(f"Debian snapshot: {snapshot} ({'updated' if changed else 'unchanged'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
