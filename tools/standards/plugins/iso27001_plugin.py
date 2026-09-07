"""ISO 27001 heuristic plugin: secrets, pinning, TLS.

Deterministic regex/walker rules. This is not an ISO 27001 certification.
"""

from __future__ import annotations

import json
import os
import re

from standards import (
    ACTION_WARN,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    StandardsPlugin,
    Violation,
)

ISO_PLUGIN_NAME = "iso27001"
ISO_VERSION = "1.0.0"
ISO_SECRET_AWS = "ISO-SECRET-AWS"
ISO_SECRET_KEY = "ISO-SECRET-KEY"
ISO_SECRET_PRIVATE_KEY = "ISO-SECRET-PRIVATE-KEY"
ISO_ENV_COMMITTED = "ISO-ENV-COMMITTED"
ISO_UNPINNED_DEP = "ISO-UNPINNED-DEP"
ISO_HTTP_INSECURE = "ISO-HTTP-INSECURE"
ISO_TLS_DISABLED = "ISO-TLS-DISABLED"

_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_API_KEY_RE = re.compile(
    r"(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{16,}['\"]",
    re.IGNORECASE,
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
_HTTP_RE = re.compile(r"https?://[^\s'\"\)]+", re.IGNORECASE)
_TLS_OFF_RE = re.compile(
    r"verify\s*=\s*False|ssl\s*=\s*False|InsecureRequestWarning|CERT_NONE",
    re.IGNORECASE,
)
_ENV_NAMES = {".env", ".env.local", ".env.production", ".env.development"}
_PINNED_REQ = re.compile(r"==|===")
_UNPINNED_JSON = re.compile(r"^[\^~*]|latest$", re.IGNORECASE)


def _violation(
    rule_id: str,
    severity: str,
    file: str,
    line: int | None,
    message: str,
    remediation: str,
) -> Violation:
    return Violation(
        plugin_name=ISO_PLUGIN_NAME,
        rule_id=rule_id,
        severity=severity,
        file=file,
        line=line,
        message=message,
        remediation=remediation,
        action=ACTION_WARN,
        fixable=False,
    )


def _is_localhost(url: str) -> bool:
    lowered = url.lower()
    return any(
        token in lowered
        for token in ("localhost", "127.0.0.1", "[::1]", "0.0.0.0", "example.com")
    )


class Iso27001Plugin(StandardsPlugin):
    name = ISO_PLUGIN_NAME
    display_name = "ISO 27001"
    description = "Heuristic secrets, dependency pinning, and TLS checks."
    version = ISO_VERSION
    severity_default = SEVERITY_WARNING

    def check(self, file: str, content: str) -> list[Violation]:
        hits: list[Violation] = []
        base = os.path.basename(file)
        if base in _ENV_NAMES:
            hits.append(
                _violation(
                    ISO_ENV_COMMITTED,
                    SEVERITY_ERROR,
                    file,
                    None,
                    "Dotenv file is present in the scanned worktree",
                    "Keep secrets out of git; gitignore .env and load from a secret store",
                )
            )
        if base in {"requirements.txt", "requirements-dev.txt"}:
            for line_no, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                    continue
                if not _PINNED_REQ.search(stripped):
                    hits.append(
                        _violation(
                            ISO_UNPINNED_DEP,
                            SEVERITY_INFO,
                            file,
                            line_no,
                            f"Unpinned Python dependency: {stripped}",
                            "Pin with == so installs are reproducible",
                        )
                    )
        if base == "package.json":
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {}
            for section in ("dependencies", "devDependencies"):
                deps = data.get(section) if isinstance(data, dict) else None
                if not isinstance(deps, dict):
                    continue
                for name, version in deps.items():
                    if isinstance(version, str) and _UNPINNED_JSON.search(version.strip()):
                        hits.append(
                            _violation(
                                ISO_UNPINNED_DEP,
                                SEVERITY_INFO,
                                file,
                                None,
                                f"Unpinned npm dependency: {name}@{version}",
                                "Pin exact versions instead of ^, ~, *, or latest",
                            )
                        )
        for line_no, line in enumerate(content.splitlines(), start=1):
            if _AWS_KEY_RE.search(line):
                hits.append(
                    _violation(
                        ISO_SECRET_AWS,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "Possible hardcoded AWS access key",
                        "Rotate the key and load credentials from a secret manager",
                    )
                )
            if _API_KEY_RE.search(line):
                hits.append(
                    _violation(
                        ISO_SECRET_KEY,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "Possible hardcoded API key or token",
                        "Move secrets to environment variables or a vault",
                    )
                )
            if _PRIVATE_KEY_RE.search(line):
                hits.append(
                    _violation(
                        ISO_SECRET_PRIVATE_KEY,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "Possible private key material in source",
                        "Remove the key from the repo and rotate it",
                    )
                )
            if _TLS_OFF_RE.search(line):
                hits.append(
                    _violation(
                        ISO_TLS_DISABLED,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "TLS verification appears disabled",
                        "Keep certificate verification enabled on remote endpoints",
                    )
                )
            for url in _HTTP_RE.findall(line):
                if url.lower().startswith("http://") and not _is_localhost(url):
                    hits.append(
                        _violation(
                            ISO_HTTP_INSECURE,
                            SEVERITY_WARNING,
                            file,
                            line_no,
                            f"Cleartext HTTP URL in a possible security context: {url}",
                            "Use HTTPS for remote endpoints",
                        )
                    )
        return hits

    def prompt_instructions(self) -> str:
        return (
            "Do not commit secrets (AWS keys, API tokens, private keys, .env files). "
            "Pin dependencies. Do not disable TLS verification. Prefer https:// for "
            f"remote endpoints ({ISO_HTTP_INSECURE}, {ISO_TLS_DISABLED})."
        )


PLUGIN = Iso27001Plugin()
