"""Cyber Essentials heuristic plugin: defaults, crypto, CORS, debug, auth.

Deterministic regex/walker rules. This is not a Cyber Essentials certification.
"""

from __future__ import annotations

import re

from standards import (
    ACTION_WARN,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    StandardsPlugin,
    Violation,
)

CE_PLUGIN_NAME = "cyber_essentials"
CE_VERSION = "1.0.0"
CE_DEFAULT_CREDS = "CE-DEFAULT-CREDS"
CE_WEAK_CRYPTO = "CE-WEAK-CRYPTO"
CE_PERMISSIVE_CORS = "CE-PERMISSIVE-CORS"
CE_DEBUG_ROUTE = "CE-DEBUG-ROUTE"
CE_DEBUG_FLAG = "CE-DEBUG-FLAG"
CE_UNBOUNDED_AUTH = "CE-UNBOUNDED-AUTH"

_DEFAULT_CREDS_RE = re.compile(
    r"(?:username|user)\s*[:=]\s*['\"]admin['\"].{0,80}(?:password|passwd)\s*[:=]\s*['\"](?:admin|password|1234)?['\"]"
    r"|(?:password|passwd)\s*[:=]\s*['\"](?:admin|password|1234)['\"]"
    r"|admin\s*/\s*admin",
    re.IGNORECASE | re.DOTALL,
)
_WEAK_CRYPTO_RE = re.compile(
    r"hashlib\.(md5|sha1)\(|Crypto\.Hash\.(MD5|SHA1)|Crypto\.Cipher\.DES|"
    r"createHash\(\s*['\"]md5['\"]|createHash\(\s*['\"]sha1['\"]|"
    r"algorithm\s*=\s*['\"](?:md5|sha1|des)['\"]",
    re.IGNORECASE,
)
_CORS_RE = re.compile(
    r"Access-Control-Allow-Origin[^\n]*\*"
    r"|allow_origins\s*=\s*\[['\"]\*['\"]\]"
    r"|cors\(\s*origins?\s*=\s*['\"]\*['\"]",
    re.IGNORECASE,
)
_DEBUG_ROUTE_RE = re.compile(
    r"""['\"]/(?:debug|__debug__)(?:/|['\"])|--debug\b|app\.run\([^)]*debug\s*=\s*True""",
    re.IGNORECASE,
)
_DEBUG_FLAG_RE = re.compile(r"\bDEBUG\s*=\s*True\b")
_ROUTE_RE = re.compile(
    r"@(?:app|router|blueprint)\.(?:route|get|post|put|delete|patch)\(|"
    r"app\.(?:get|post|put|delete)\(",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"\b(login_required|requires_auth|auth_required|Depends\(|middleware|"
    r"Authorization|Bearer|session\[|current_user)\b",
    re.IGNORECASE,
)


def _violation(
    rule_id: str,
    severity: str,
    file: str,
    line: int | None,
    message: str,
    remediation: str,
    *,
    fixable: bool = False,
) -> Violation:
    return Violation(
        plugin_name=CE_PLUGIN_NAME,
        rule_id=rule_id,
        severity=severity,
        file=file,
        line=line,
        message=message,
        remediation=remediation,
        action=ACTION_WARN,
        fixable=fixable,
    )


class CyberEssentialsPlugin(StandardsPlugin):
    name = CE_PLUGIN_NAME
    display_name = "Cyber Essentials"
    description = "Heuristic default-credential, crypto, CORS, debug, and auth checks."
    version = CE_VERSION
    severity_default = SEVERITY_WARNING

    def check(self, file: str, content: str) -> list[Violation]:
        hits: list[Violation] = []
        if _DEFAULT_CREDS_RE.search(content):
            line_no = next(
                (
                    index
                    for index, line in enumerate(content.splitlines(), start=1)
                    if _DEFAULT_CREDS_RE.search(line)
                    or "admin/admin" in line.lower()
                ),
                None,
            )
            hits.append(
                _violation(
                    CE_DEFAULT_CREDS,
                    SEVERITY_ERROR,
                    file,
                    line_no,
                    "Default or empty credential pattern",
                    "Do not ship admin/admin, password=password, or empty default secrets",
                )
            )
        for line_no, line in enumerate(content.splitlines(), start=1):
            if _WEAK_CRYPTO_RE.search(line):
                hits.append(
                    _violation(
                        CE_WEAK_CRYPTO,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "Weak hash or cipher used in a possible auth/integrity context",
                        "Use SHA-256+ or a password KDF; never MD5, SHA-1, or DES for auth",
                    )
                )
            if _CORS_RE.search(line):
                hits.append(
                    _violation(
                        CE_PERMISSIVE_CORS,
                        SEVERITY_WARNING,
                        file,
                        line_no,
                        "Permissive CORS origin *",
                        "Restrict Access-Control-Allow-Origin to known hosts",
                    )
                )
            if _DEBUG_ROUTE_RE.search(line):
                hits.append(
                    _violation(
                        CE_DEBUG_ROUTE,
                        SEVERITY_WARNING,
                        file,
                        line_no,
                        "Debug route or debug server flag",
                        "Do not ship /debug endpoints or debug servers in production",
                    )
                )
            if _DEBUG_FLAG_RE.search(line):
                hits.append(
                    _violation(
                        CE_DEBUG_FLAG,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "DEBUG = True would ship a debug configuration",
                        "Set DEBUG = False outside local development",
                        fixable=True,
                    )
                )
        if _ROUTE_RE.search(content) and not _AUTH_RE.search(content):
            hits.append(
                _violation(
                    CE_UNBOUNDED_AUTH,
                    SEVERITY_INFO,
                    file,
                    None,
                    "HTTP routes without an auth middleware/decorator heuristic",
                    "Gate mutating and private routes behind authentication",
                )
            )
        return hits

    def prompt_instructions(self) -> str:
        return (
            "No default credentials. Do not use MD5, SHA-1, or DES for auth. "
            "Do not allow CORS origin *. Do not ship /debug routes or DEBUG = True. "
            f"Protect HTTP routes with auth ({CE_UNBOUNDED_AUTH})."
        )

    def auto_fix(self, file: str, content: str) -> str | None:
        if not _DEBUG_FLAG_RE.search(content):
            return None
        return _DEBUG_FLAG_RE.sub("DEBUG = False", content)


PLUGIN = CyberEssentialsPlugin()
