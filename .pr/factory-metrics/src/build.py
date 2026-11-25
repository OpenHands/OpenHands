#!/usr/bin/env python3
"""Build ../index.html (self-contained, no script) from index.html.

index.html uses a small script to draw the sparklines and bar charts. This
build renders the page in headless Chromium, keeps the resulting static SVG and
removes the build-time script. htmlpreview.github.io then shows the
page without running anything, which is the `.pr/` design-doc convention.

Usage:  python3 .pr/factory-metrics/src/build.py
Needs:  chromium, chromium-browser or google-chrome on PATH.
"""
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC, OUT = HERE / "index.html", HERE.parent / "index.html"

browser = next(
    (b for b in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable") if shutil.which(b)),
    None,
)
if browser is None:
    sys.exit("Install Chromium or Chrome: it renders the SVG charts.")

dom = subprocess.run(
    [
        browser, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
        "--window-size=1600,2000", "--virtual-time-budget=4000", "--dump-dom", SRC.as_uri(),
    ],
    check=True, capture_output=True, text=True,
).stdout

dom = re.sub(r"<script\b.*?</script>", "", dom, flags=re.S)
assert "<script" not in dom and "<link" not in dom, "build left a script or external stylesheet in the output"
OUT.write_text("<!doctype html>\n" + dom)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")
