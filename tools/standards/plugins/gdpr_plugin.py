"""GDPR heuristic plugin: PII patterns, logging, consent, erasure, storage.

Deterministic regex/walker rules. This is not a GDPR certification.
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

GDPR_PLUGIN_NAME = "gdpr"
GDPR_VERSION = "1.0.0"
GDPR_EMAIL = "GDPR-EMAIL"
GDPR_PHONE = "GDPR-PHONE"
GDPR_PERSONAL_ID = "GDPR-PERSONAL-ID"
GDPR_PII_LOG = "GDPR-PII-LOG"
GDPR_CONSENT = "GDPR-CONSENT"
GDPR_ERASURE = "GDPR-ERASURE"
GDPR_UNENCRYPTED = "GDPR-UNENCRYPTED"

_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_EXAMPLE_DOMAINS = (
    "example.com",
    "example.org",
    "example.net",
    "test.local",
    "localhost",
)
_PHONE_RE = re.compile(
    r"\+(?:33|34|44|49|1)[\s.-]?(?:\(?\d{2,3}\)?[\s.-]?){2,4}\d{2,4}"
)
_PESEL_RE = re.compile(r"\bPESEL[:\s]+\d{11}\b", re.IGNORECASE)
_DNI_RE = re.compile(r"\bDNI[:\s]+\d{8}[A-Z]\b", re.IGNORECASE)
_NINO_RE = re.compile(r"\bNINO[:\s]+[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", re.IGNORECASE)
_LOG_RE = re.compile(
    r"(?:print|logger?\.(?:info|debug|warning|error)|console\.log)\s*\(",
    re.IGNORECASE,
)
_EMAIL_FIELD_RE = re.compile(
    r"type\s*=\s*[\"']email[\"']|name\s*=\s*[\"']e-?mail[\"']",
    re.IGNORECASE,
)
_CONSENT_RE = re.compile(r"\b(consent|opt[-_ ]?in|gdpr)\b", re.IGNORECASE)
_ERASURE_RE = re.compile(
    r"\b(erasure|right to be forgotten|/delete|/users/.+delete|forget[_-]?me)\b",
    re.IGNORECASE,
)
_UNENCRYPTED_RE = re.compile(
    r"pickle\.dump\(|json\.dump\(|sqlite3\.connect\(",
    re.IGNORECASE,
)
_USER_RECORD_RE = re.compile(
    r"\b(user_record|users?|customer|pii|personal)\b",
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
        plugin_name=GDPR_PLUGIN_NAME,
        rule_id=rule_id,
        severity=severity,
        file=file,
        line=line,
        message=message,
        remediation=remediation,
        action=ACTION_WARN,
        fixable=fixable,
    )


def _real_emails(text: str) -> list[str]:
    found: list[str] = []
    for match in _EMAIL_RE.finditer(text):
        value = match.group(0)
        domain = value.rsplit("@", 1)[-1].lower()
        if any(domain == item or domain.endswith(f".{item}") for item in _EXAMPLE_DOMAINS):
            continue
        found.append(value)
    return found


class GdprPlugin(StandardsPlugin):
    name = GDPR_PLUGIN_NAME
    display_name = "GDPR"
    description = "Heuristic PII, consent, erasure, and storage checks."
    version = GDPR_VERSION
    severity_default = SEVERITY_WARNING

    def check(self, file: str, content: str) -> list[Violation]:
        hits: list[Violation] = []
        emails = _real_emails(content)
        lower = file.lower()
        collects = bool(_EMAIL_FIELD_RE.search(content)) or bool(emails)
        for line_no, line in enumerate(content.splitlines(), start=1):
            line_emails = _real_emails(line)
            if line_emails:
                hits.append(
                    _violation(
                        GDPR_EMAIL,
                        SEVERITY_WARNING,
                        file,
                        line_no,
                        f"Possible personal email address: {line_emails[0]}",
                        "Do not hard-code personal emails; use fixtures like user@example.com",
                    )
                )
            if _PHONE_RE.search(line):
                hits.append(
                    _violation(
                        GDPR_PHONE,
                        SEVERITY_WARNING,
                        file,
                        line_no,
                        "Possible personal phone number",
                        "Remove live phone numbers from source and logs",
                    )
                )
            if _PESEL_RE.search(line) or _DNI_RE.search(line) or _NINO_RE.search(line):
                hits.append(
                    _violation(
                        GDPR_PERSONAL_ID,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "Possible national personal identifier",
                        "Never store national IDs in source or plaintext logs",
                    )
                )
            if _LOG_RE.search(line) and (
                line_emails or _PHONE_RE.search(line) or "email" in line.lower()
            ):
                hits.append(
                    _violation(
                        GDPR_PII_LOG,
                        SEVERITY_ERROR,
                        file,
                        line_no,
                        "Possible logging or printing of personal data",
                        "Redact PII from logs and stdout",
                    )
                )
            if _UNENCRYPTED_RE.search(line) and _USER_RECORD_RE.search(line):
                hits.append(
                    _violation(
                        GDPR_UNENCRYPTED,
                        SEVERITY_WARNING,
                        file,
                        line_no,
                        "Unencrypted storage of data that may include personal records",
                        "Encrypt personal data at rest and avoid pickle of user records",
                    )
                )
        if collects and not _CONSENT_RE.search(content):
            hits.append(
                _violation(
                    GDPR_CONSENT,
                    SEVERITY_WARNING,
                    file,
                    None,
                    "Collects email-like personal data without a consent/opt-in marker",
                    "Add an explicit opt-in/consent control next to the collection point",
                )
            )
        if collects and lower.endswith((".py", ".js", ".ts", ".tsx", ".go")):
            if not _ERASURE_RE.search(content):
                hits.append(
                    _violation(
                        GDPR_ERASURE,
                        SEVERITY_INFO,
                        file,
                        None,
                        "Personal-data collection without an erasure/deletion route in this file",
                        "Expose a documented deletion or erasure endpoint for stored personal data",
                    )
                )
        return hits

    def prompt_instructions(self) -> str:
        return (
            "Follow GDPR data-minimisation: never hard-code personal emails, phone "
            "numbers, or national IDs; do not log PII; collect email only with an "
            f"explicit opt-in ({GDPR_CONSENT}); provide an erasure/deletion path "
            f"({GDPR_ERASURE}); do not pickle or dump personal records unencrypted."
        )


PLUGIN = GdprPlugin()
