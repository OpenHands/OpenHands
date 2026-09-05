#!/usr/bin/env python3
"""Regression checks for the agent-canvas Helm chart.

The tests intentionally use only the Python standard library plus the Helm CLI
so they can run in CI without installing a Helm test plugin or YAML parser.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


CHART_DIR = Path(__file__).resolve().parents[1]


def run_helm(*args: str) -> str:
    result = subprocess.run(
        ["helm", *args],
        check=True,
        cwd=CHART_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def render(*args: str) -> str:
    return run_helm(
        "template",
        "canvas",
        str(CHART_DIR),
        "--namespace",
        "agents",
        *args,
    )


def documents(manifest: str, kind: str) -> list[str]:
    return [
        document
        for document in re.split(r"^---\s*$", manifest, flags=re.MULTILINE)
        if re.search(rf"^kind: {re.escape(kind)}\s*$", document, re.MULTILINE)
    ]


def one_document(manifest: str, kind: str) -> str:
    matches = documents(manifest, kind)
    assert len(matches) == 1, f"expected one {kind}, found {len(matches)}"
    return matches[0]


def assert_default_render() -> None:
    manifest = render()
    deployment = one_document(manifest, "Deployment")
    pvc = one_document(manifest, "PersistentVolumeClaim")

    assert not documents(manifest, "StatefulSet")
    assert len(documents(manifest, "Service")) == 1
    assert "name: canvas-agent-canvas-data" in pvc
    assert "helm.sh/resource-policy: keep" in pvc
    assert "storage: \"20Gi\"" in pvc
    assert "strategy:\n    type: Recreate" in deployment
    assert "claimName: canvas-agent-canvas-data" in deployment
    assert deployment.count("- name: data") == 3


def assert_existing_claim_render() -> None:
    manifest = render(
        "--set-string", "persistence.existingClaim=existing.production.claim"
    )
    deployment = one_document(manifest, "Deployment")

    assert not documents(manifest, "PersistentVolumeClaim")
    assert "claimName: existing.production.claim" in deployment
    assert "- name: existing.production.claim" not in deployment
    assert deployment.count("- name: data") == 3


def assert_persistence_disabled_render() -> None:
    manifest = render("--set", "persistence.enabled=false")
    deployment = one_document(manifest, "Deployment")

    assert not documents(manifest, "PersistentVolumeClaim")
    assert "volumeMounts:" not in deployment
    assert "volumes:" not in deployment


def assert_custom_pvc_and_strategy_render() -> None:
    manifest = render(
        "--set-string",
        "persistence.storageClassName=fast-rwo",
        "--set-string",
        "persistence.size=50Gi",
        "--set-string",
        r"persistence.annotations.backup\.example\.com/enabled=true",
        "--set",
        "deployment.strategy.type=RollingUpdate",
        "--set",
        "deployment.strategy.rollingUpdate.maxUnavailable=1",
        "--set",
        "deployment.strategy.rollingUpdate.maxSurge=0",
    )
    deployment = one_document(manifest, "Deployment")
    pvc = one_document(manifest, "PersistentVolumeClaim")

    assert "storageClassName: \"fast-rwo\"" in pvc
    assert "storage: \"50Gi\"" in pvc
    assert "backup.example.com/enabled: \"true\"" in pvc
    assert "type: RollingUpdate" in deployment
    assert "maxUnavailable: 1" in deployment
    assert "maxSurge: 0" in deployment


def assert_missing_deployment_values_render() -> None:
    # Helm's --reuse-values can omit newly introduced defaults during an
    # upgrade from chart 0.1.x. Keep the safe single-writer strategy anyway.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as values_file:
        values_file.write("deployment: null\n")
        values_file.flush()
        deployment = one_document(render("--values", values_file.name), "Deployment")

    assert "strategy:\n    type: Recreate" in deployment


def assert_notes_target_deployment() -> None:
    notes = (CHART_DIR / "templates" / "NOTES.txt").read_text()
    assert "rollout status deployment/" in notes
    assert "statefulset/" not in notes.lower()
    assert "/home/openhands/.openhands/agent-canvas/api-key.txt" in notes


def main() -> None:
    run_helm("lint", str(CHART_DIR))
    assert_default_render()
    assert_existing_claim_render()
    assert_persistence_disabled_render()
    assert_custom_pvc_and_strategy_render()
    assert_missing_deployment_values_render()
    assert_notes_target_deployment()
    print("agent-canvas Helm render tests passed")


if __name__ == "__main__":
    main()
