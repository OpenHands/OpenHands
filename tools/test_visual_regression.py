"""Integration tests for the visual-regression loop.

Run from the repo root:

    python3 tools/test_visual_regression.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from loop_runner import LoopStore, STATUS_FAILED, STATUS_PASSED  # noqa: E402
from visual_regression import (  # noqa: E402
    DEFAULT_THRESHOLD,
    VISUAL_REGRESSION_NAME,
    VisualRegressionService,
    diff_ratio,
)
from visual_regression_api import handle_request  # noqa: E402


def _png(path: str, color: tuple[int, int, int], mark: tuple[int, int] | None = None) -> None:
    from PIL import Image

    image = Image.new("RGB", (4, 4), color)
    if mark is not None:
        image.putpixel(mark, (255, 255, 255))
    image.save(path)


class VisualRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp()
        self.assets = os.path.join(self.workdir, "assets")
        self.loops = LoopStore(":memory:")
        self.service = VisualRegressionService(
            loop_store=self.loops, assets_root=self.assets
        )
        self.baseline_png = os.path.join(self.workdir, "baseline.png")
        self.changed_png = os.path.join(self.workdir, "changed.png")
        _png(self.baseline_png, (0, 0, 0))
        _png(self.changed_png, (0, 0, 0), mark=(1, 1))
        self._source = self.baseline_png

        def fake_capture(url: str, width: int, height: int, dest_path: str) -> None:
            del url, width, height
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            from PIL import Image

            Image.open(self._source).save(dest_path)

        self._orig_capture = __import__(
            "visual_regression", fromlist=["capture_screenshot"]
        ).capture_screenshot
        import visual_regression as module

        module.capture_screenshot = fake_capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        import visual_regression as module

        module.capture_screenshot = self._orig_capture  # type: ignore[assignment]
        self.service.close()
        self.loops.close()

    def test_diff_ratio_math(self) -> None:
        ratio = diff_ratio(self.changed_png, self.baseline_png, pixel_tolerance=0)
        self.assertAlmostEqual(ratio, 1 / 16)
        same = diff_ratio(self.baseline_png, self.baseline_png, pixel_tolerance=0)
        self.assertEqual(same, 0.0)

    def test_threshold_pass_fail_and_baseline(self) -> None:
        definition = self.service.setup(
            "proj-1",
            {
                "urls": ["http://example.test/home"],
                "viewports": [{"width": 4, "height": 4}],
                "threshold": 0.002,
            },
        )
        self.assertEqual(definition["name"], VISUAL_REGRESSION_NAME)
        again = self.service.setup("proj-1", definition["config"])
        self.assertEqual(again["id"], definition["id"])

        first = self.loops.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(first["status"], STATUS_PASSED)
        artifacts = self.service.artifacts(first["id"])
        self.assertTrue(artifacts["baseline_updated"])
        baseline = artifacts["images"][0]["baseline"]
        self.assertTrue(os.path.isfile(baseline))
        capture = artifacts["images"][0]["capture"]
        self.assertTrue(capture.startswith(os.path.join(self.assets, first["id"])))

        self._source = self.changed_png
        failed = self.loops.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(failed["status"], STATUS_FAILED)
        failed_art = self.service.artifacts(failed["id"])
        self.assertFalse(failed_art["baseline_updated"])
        self.assertGreater(failed_art["images"][0]["diff_ratio"], DEFAULT_THRESHOLD)
        from PIL import Image

        self.assertEqual(Image.open(baseline).getpixel((1, 1)), (0, 0, 0))

        self.service.setup(
            "proj-1",
            {
                "urls": ["http://example.test/home"],
                "viewports": [{"width": 4, "height": 4}],
                "threshold": 0.1,
            },
        )
        passed = self.loops.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(passed["status"], STATUS_PASSED)
        self.assertTrue(self.service.artifacts(passed["id"])["baseline_updated"])

    def test_capture_baseline_and_artifacts_api(self) -> None:
        status, definition = handle_request(
            self.service,
            "POST",
            "/api/visual-regression/projects/proj-1/setup",
            {
                "urls": ["http://example.test/home"],
                "viewports": [{"width": 4, "height": 4}],
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(definition["name"], VISUAL_REGRESSION_NAME)
        status, run = handle_request(
            self.service,
            "POST",
            "/api/visual-regression/projects/proj-1/capture-baseline",
            {"worktree_dir": self.workdir},
        )
        self.assertEqual(status, 200)
        self.assertEqual(run["status"], STATUS_PASSED)
        status, artifacts = handle_request(
            self.service,
            "GET",
            f"/api/visual-regression/runs/{run['id']}/artifacts",
        )
        self.assertEqual(status, 200)
        self.assertTrue(artifacts["baseline_updated"])
        self.assertEqual(len(artifacts["images"]), 1)


if __name__ == "__main__":
    unittest.main()
