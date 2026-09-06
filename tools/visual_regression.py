"""Visual regression loop: Playwright capture + Pillow pixel diff.

Stages ``capture`` and ``compare`` register as loop-runner handlers so tests
can monkeypatch the browser call. Screenshots live under
``~/.openhands/agent-canvas/assets/loops/<run_id>/``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any
from urllib.parse import urlparse

from loop_runner import (
    LoopStore,
    ON_FAILURE_STOP,
    STATUS_PASSED,
    register_stage_handler,
)

VISUAL_REGRESSION_NAME = "visual-regression"
VISUAL_STAGES: list[dict[str, Any]] = [
    {"name": "capture", "cmd": None, "iterative": False},
    {"name": "compare", "cmd": None, "iterative": False},
]
DEFAULT_THRESHOLD = 0.002
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
PIXEL_TOLERANCE = 8
PLAYWRIGHT_HINT = "playwright install chromium"
MANIFEST_FILENAME = "manifest.json"
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class VisualRegressionError(Exception):
    """Raised for invalid visual-regression operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def default_assets_root() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    return os.path.join(root, "assets", "loops")


def image_slug(url: str, width: int, height: int) -> str:
    parsed = urlparse(url)
    host = _SLUG_RE.sub("-", parsed.netloc or "local").strip("-.").lower() or "local"
    path = _SLUG_RE.sub("-", parsed.path.strip("/") or "index").strip("-.").lower() or "index"
    return f"{host}-{path}_{width}x{height}"


def capture_screenshot(url: str, width: int, height: int, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise VisualRegressionError(
            f"playwright is required; pip install playwright && {PLAYWRIGHT_HINT}"
        ) from exc
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(url)
                page.screenshot(path=dest_path)
            finally:
                browser.close()
    except VisualRegressionError:
        raise
    except Exception as exc:
        raise VisualRegressionError(
            f"playwright capture failed ({exc}); run: {PLAYWRIGHT_HINT}"
        ) from exc


def diff_ratio(
    current_path: str,
    baseline_path: str,
    pixel_tolerance: int = PIXEL_TOLERANCE,
) -> float:
    from PIL import Image, ImageChops

    current = Image.open(current_path).convert("RGB")
    baseline = Image.open(baseline_path).convert("RGB")
    if current.size != baseline.size:
        current = current.resize(baseline.size)
    diff = ImageChops.difference(current, baseline)
    width, height = current.size
    total = width * height
    pixels = diff.load()
    changed = 0
    for y in range(height):
        for x in range(width):
            if max(pixels[x, y]) > pixel_tolerance:
                changed += 1
    return 0.0 if total == 0 else changed / total


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(config or {})
    urls = payload.get("urls") or []
    if not isinstance(urls, list):
        raise VisualRegressionError("config.urls must be a list")
    viewports = payload.get("viewports") or [dict(DEFAULT_VIEWPORT)]
    if not isinstance(viewports, list) or not viewports:
        raise VisualRegressionError("config.viewports must be a non-empty list")
    normalized_viewports: list[dict[str, int]] = []
    for viewport in viewports:
        if not isinstance(viewport, dict):
            raise VisualRegressionError("each viewport must be an object")
        width = int(viewport.get("width") or 0)
        height = int(viewport.get("height") or 0)
        if width < 1 or height < 1:
            raise VisualRegressionError("viewport width and height must be >= 1")
        normalized_viewports.append({"width": width, "height": height})
    threshold = float(payload.get("threshold") if payload.get("threshold") is not None else DEFAULT_THRESHOLD)
    if threshold < 0:
        raise VisualRegressionError("threshold must be >= 0")
    baseline_dir = payload.get("baseline_dir")
    return {
        "urls": [str(url) for url in urls],
        "viewports": normalized_viewports,
        "threshold": threshold,
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
    }


class VisualRegressionService:
    """Registers and runs the visual-regression loop for a project."""

    def __init__(
        self,
        loop_store: LoopStore | None = None,
        assets_root: str | None = None,
    ) -> None:
        self.loop_store = loop_store or LoopStore()
        self.assets_root = assets_root or default_assets_root()
        self._owns_loop_store = loop_store is None
        register_stage_handler(VISUAL_REGRESSION_NAME, "capture", self._handle_capture)
        register_stage_handler(VISUAL_REGRESSION_NAME, "compare", self._handle_compare)

    def close(self) -> None:
        if self._owns_loop_store:
            self.loop_store.close()

    def setup(self, project_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        if not project_id:
            raise VisualRegressionError("project_id is required")
        normalized = normalize_config(config)
        existing = self._definition_for(project_id)
        if existing is not None:
            return self.loop_store.set_definition_config(existing["id"], normalized)
        return self.loop_store.create_definition(
            name=VISUAL_REGRESSION_NAME,
            project_id=project_id,
            stages=VISUAL_STAGES,
            max_iterations=1,
            on_failure=ON_FAILURE_STOP,
            config=normalized,
        )

    def capture_baseline(
        self, project_id: str, worktree_dir: str
    ) -> dict[str, Any]:
        definition = self.setup(project_id, self._config_for(project_id))
        run = self.loop_store.start_run(definition["id"], worktree_dir=worktree_dir)
        if run["status"] != STATUS_PASSED:
            return run
        self._promote_baseline(run["id"], definition, force=True)
        manifest = self._read_manifest(run["id"])
        manifest["baseline_updated"] = True
        self._write_manifest(run["id"], manifest)
        return run

    def artifacts(self, run_id: str) -> dict[str, Any]:
        run = self.loop_store.get_run(run_id)
        manifest = self._read_manifest(run_id)
        return {
            "run_id": run_id,
            "status": run["status"],
            "images": manifest.get("images") or [],
            "baseline_updated": bool(manifest.get("baseline_updated")),
            "threshold": manifest.get("threshold"),
        }

    def _definition_for(self, project_id: str) -> dict[str, Any] | None:
        for definition in self.loop_store.list_definitions():
            if (
                definition["project_id"] == project_id
                and definition["name"] == VISUAL_REGRESSION_NAME
            ):
                return definition
        return None

    def _config_for(self, project_id: str) -> dict[str, Any]:
        definition = self._definition_for(project_id)
        return normalize_config(definition["config"] if definition else None)

    def _run_dir(self, run_id: str) -> str:
        path = os.path.join(self.assets_root, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _baseline_dir(self, definition: dict[str, Any]) -> str:
        config = normalize_config(definition.get("config"))
        if config["baseline_dir"]:
            path = config["baseline_dir"]
        else:
            path = os.path.join(
                self.assets_root, "baselines", definition["project_id"]
            )
        os.makedirs(path, exist_ok=True)
        return path

    def _targets(self, config: dict[str, Any]) -> list[tuple[str, dict[str, int], str]]:
        if not config["urls"]:
            raise VisualRegressionError("config.urls is empty")
        targets: list[tuple[str, dict[str, int], str]] = []
        for url in config["urls"]:
            for viewport in config["viewports"]:
                slug = image_slug(url, viewport["width"], viewport["height"])
                targets.append((url, viewport, slug))
        return targets

    def _handle_capture(
        self,
        run_id: str,
        stage: dict[str, Any],
        worktree_dir: str,
        definition: dict[str, Any],
    ) -> tuple[bool, str]:
        del stage, worktree_dir
        config = normalize_config(definition.get("config"))
        run_dir = self._run_dir(run_id)
        images: list[dict[str, Any]] = []
        for url, viewport, slug in self._targets(config):
            dest = os.path.join(run_dir, f"{slug}.png")
            capture_screenshot(url, viewport["width"], viewport["height"], dest)
            images.append(
                {
                    "slug": slug,
                    "url": url,
                    "viewport": viewport,
                    "capture": dest,
                    "baseline": None,
                    "diff_ratio": None,
                    "passed": None,
                }
            )
        self._write_manifest(
            run_id,
            {
                "images": images,
                "baseline_updated": False,
                "threshold": config["threshold"],
            },
        )
        return True, json.dumps({"captured": len(images)})

    def _handle_compare(
        self,
        run_id: str,
        stage: dict[str, Any],
        worktree_dir: str,
        definition: dict[str, Any],
    ) -> tuple[bool, str]:
        del stage, worktree_dir
        config = normalize_config(definition.get("config"))
        manifest = self._read_manifest(run_id)
        images = list(manifest.get("images") or [])
        baseline_dir = self._baseline_dir(definition)
        first_run = not any(
            os.path.isfile(os.path.join(baseline_dir, f"{item['slug']}.png"))
            for item in images
        )
        failed = False
        for item in images:
            capture_path = item["capture"]
            baseline_path = os.path.join(baseline_dir, f"{item['slug']}.png")
            item["baseline"] = baseline_path
            if first_run or not os.path.isfile(baseline_path):
                item["diff_ratio"] = 0.0
                item["passed"] = True
                continue
            ratio = diff_ratio(capture_path, baseline_path)
            item["diff_ratio"] = ratio
            item["passed"] = ratio <= config["threshold"]
            if not item["passed"]:
                failed = True
        baseline_updated = False
        if first_run or not failed:
            self._promote_baseline(run_id, definition, force=True)
            baseline_updated = True
        manifest["images"] = images
        manifest["baseline_updated"] = baseline_updated
        manifest["threshold"] = config["threshold"]
        self._write_manifest(run_id, manifest)
        summary = json.dumps(
            {
                "failed": failed,
                "baseline_updated": baseline_updated,
                "images": [
                    {"slug": item["slug"], "diff_ratio": item["diff_ratio"]}
                    for item in images
                ],
            }
        )
        return (not failed), summary

    def _promote_baseline(
        self, run_id: str, definition: dict[str, Any], force: bool = False
    ) -> None:
        del force
        manifest = self._read_manifest(run_id)
        baseline_dir = self._baseline_dir(definition)
        for item in manifest.get("images") or []:
            capture_path = item.get("capture")
            if not capture_path or not os.path.isfile(capture_path):
                continue
            dest = os.path.join(baseline_dir, f"{item['slug']}.png")
            shutil.copy2(capture_path, dest)
            item["baseline"] = dest
        self._write_manifest(run_id, manifest)

    def _manifest_path(self, run_id: str) -> str:
        return os.path.join(self._run_dir(run_id), MANIFEST_FILENAME)

    def _read_manifest(self, run_id: str) -> dict[str, Any]:
        path = self._manifest_path(run_id)
        if not os.path.isfile(path):
            return {"images": [], "baseline_updated": False}
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def _write_manifest(self, run_id: str, manifest: dict[str, Any]) -> None:
        with open(self._manifest_path(run_id), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
