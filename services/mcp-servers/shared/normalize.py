"""Finding payload normalization and scope allowlist (fail-closed)."""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Literal
from urllib.parse import urlparse

Severity = Literal["critical", "high", "medium", "low", "info"]

SOURCE_TOOLS = frozenset(
    {
        "nuclei",
        "zap",
        "wapiti",
        "nikto",
        "sqlmap",
        "subfinder",
        "httpx",
        "reconftw",
    }
)

SCOPE_ALLOWLIST_ENV = "PENTEST_SCOPE_ALLOWLIST"


class ScopeViolationError(Exception):
    """Target is outside PENTEST_SCOPE_ALLOWLIST."""

    code = "scope_violation"

    def __init__(self, target: str, message: str | None = None):
        self.target = target
        super().__init__(message or f"Target out of scope: {target}")

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "target": self.target, "message": str(self)}


def normalize_finding(
    *,
    engagement_id: str,
    source_tool: str,
    title: str,
    severity: Severity,
    description: str | None = None,
    asset: str | None = None,
    endpoint: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_tool not in SOURCE_TOOLS:
        raise ValueError(f"Unsupported source_tool: {source_tool}")
    if not title.strip():
        raise ValueError("title is required")
    payload: dict[str, Any] = {
        "engagement_id": engagement_id,
        "source_tool": source_tool,
        "title": title.strip(),
        "severity": severity,
        "description": description,
        "asset": asset,
        "endpoint": endpoint,
        "evidence": evidence or {},
    }
    return payload


def _parse_allowlist() -> list[str]:
    raw = os.environ.get(SCOPE_ALLOWLIST_ENV)
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def extract_host(target: str) -> str:
    """Best-effort host extraction from URL, host:port, or bare host."""
    value = target.strip()
    if "://" in value:
        host = urlparse(value).hostname
        return (host or value).lower()
    # strip path/query if someone passed host/path
    value = value.split("/")[0]
    if value.count(":") == 1 and not value.startswith("["):
        host, _, port = value.partition(":")
        if port.isdigit():
            return host.lower()
    return value.lower()


def _host_matches(pattern: str, host: str) -> bool:
    pat = pattern.lower().strip()
    host = host.lower().strip()
    if not pat:
        return False
    # CIDR
    if "/" in pat:
        try:
            network = ipaddress.ip_network(pat, strict=False)
            return ipaddress.ip_address(host) in network
        except ValueError:
            return False
    # Exact IP or hostname
    if pat == host:
        return True
    # Wildcard DNS: *.example.com
    if pat.startswith("*."):
        suffix = pat[1:]  # .example.com
        return host.endswith(suffix) and host != pat[2:]
    # Domain suffix match: example.com matches a.example.com
    if host == pat or host.endswith("." + pat):
        return True
    return False


def assert_in_scope(target: str) -> None:
    """
    Fail-closed: empty/missing PENTEST_SCOPE_ALLOWLIST rejects all targets.
    """
    allowlist = _parse_allowlist()
    if not allowlist:
        raise ScopeViolationError(
            target,
            f"{SCOPE_ALLOWLIST_ENV} is empty or unset (fail-closed)",
        )
    host = extract_host(target)
    if any(_host_matches(entry, host) for entry in allowlist):
        return
    # Also allow exact full-target match (URL allowlist entries)
    if any(target.startswith(entry) or entry == target for entry in allowlist):
        return
    raise ScopeViolationError(target)


def assert_targets_in_scope(targets: list[str]) -> None:
    for target in targets:
        assert_in_scope(target)
