"""AC-189-A4 — Semgrep/Trivy severity → Findings enum."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from shared.normalize import (  # noqa: E402
    map_semgrep_severity,
    map_trivy_severity,
    normalize_finding,
)
from tools.semgrep_scan import findings_from_semgrep  # noqa: E402
from tools.trivy_scan import findings_from_trivy  # noqa: E402


# @spec PROJETOSIN-189 — AC-189-A4
def test_map_semgrep_severity():
    assert map_semgrep_severity("ERROR") == "high"
    assert map_semgrep_severity("WARNING") == "medium"
    assert map_semgrep_severity("INFO") == "info"
    assert map_semgrep_severity("CRITICAL") == "critical"
    assert map_semgrep_severity(None) == "info"


# @spec PROJETOSIN-189 — AC-189-A4
def test_map_trivy_severity():
    assert map_trivy_severity("CRITICAL") == "critical"
    assert map_trivy_severity("HIGH") == "high"
    assert map_trivy_severity("MEDIUM") == "medium"
    assert map_trivy_severity("LOW") == "low"
    assert map_trivy_severity("UNKNOWN") == "info"
    assert map_trivy_severity(None) == "info"


# @spec PROJETOSIN-189 — AC-189-A4
def test_normalize_semgrep_payload_from_report():
    report = {
        "results": [
            {
                "check_id": "x.y",
                "path": "a.py",
                "start": {"line": 3},
                "extra": {"message": "msg", "severity": "ERROR"},
            }
        ]
    }
    payloads = findings_from_semgrep(
        engagement_id="00000000-0000-0000-0000-000000000099",
        report=report,
    )
    assert len(payloads) == 1
    assert payloads[0]["severity"] == "high"
    assert payloads[0]["source_tool"] == "semgrep"


# @spec PROJETOSIN-189 — AC-189-A4
def test_normalize_trivy_payload_from_report():
    report = {
        "Results": [
            {
                "Target": "img",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-1",
                        "Severity": "CRITICAL",
                        "Title": "t",
                    }
                ],
            }
        ]
    }
    payloads = findings_from_trivy(
        engagement_id="00000000-0000-0000-0000-000000000099",
        report=report,
    )
    assert len(payloads) == 1
    assert payloads[0]["severity"] == "critical"
    assert payloads[0]["source_tool"] == "trivy"


def test_normalize_finding_rejects_unknown_tool():
    try:
        normalize_finding(
            engagement_id="e",
            source_tool="unknown-tool",
            title="t",
            severity="info",
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
