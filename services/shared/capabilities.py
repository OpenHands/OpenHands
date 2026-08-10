"""Pentest RBAC profiles and capabilities (mirror of src/types/pentest-rbac.ts)."""

from __future__ import annotations

from typing import Literal

PentestCapability = Literal[
    "pentest.workspace.create",
    "pentest.engagement.create",
    "pentest.engagement.view",
    "pentest.recon.run",
    "pentest.scan.passive",
    "pentest.scan.active",
    "pentest.sast.run",
    "pentest.exploit.active",
    "pentest.findings.view",
    "pentest.findings.triage",
    "pentest.findings.export_dd",
    "pentest.mobile.dynamic",
    "pentest.autonomy.autonomous",
    "pentest.admin.users",
    "pentest.admin.scope",
]

PentestProfile = Literal["admin", "pentester", "analyst", "client"]

ALL_CAPABILITIES: list[PentestCapability] = [
    "pentest.workspace.create",
    "pentest.engagement.create",
    "pentest.engagement.view",
    "pentest.recon.run",
    "pentest.scan.passive",
    "pentest.scan.active",
    "pentest.sast.run",
    "pentest.exploit.active",
    "pentest.findings.view",
    "pentest.findings.triage",
    "pentest.findings.export_dd",
    "pentest.mobile.dynamic",
    "pentest.autonomy.autonomous",
    "pentest.admin.users",
    "pentest.admin.scope",
]

PROFILE_CAPABILITIES: dict[PentestProfile, list[PentestCapability]] = {
    "admin": list(ALL_CAPABILITIES),
    "pentester": [
        "pentest.workspace.create",
        "pentest.engagement.create",
        "pentest.engagement.view",
        "pentest.recon.run",
        "pentest.scan.passive",
        "pentest.scan.active",
        "pentest.sast.run",
        "pentest.exploit.active",
        "pentest.findings.view",
        "pentest.findings.triage",
        "pentest.findings.export_dd",
        "pentest.mobile.dynamic",
        "pentest.autonomy.autonomous",
    ],
    "analyst": [
        "pentest.engagement.view",
        "pentest.findings.view",
        "pentest.findings.triage",
    ],
    "client": [
        "pentest.engagement.view",
        "pentest.findings.view",
    ],
}
