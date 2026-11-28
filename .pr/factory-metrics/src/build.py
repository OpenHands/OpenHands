#!/usr/bin/env python3
"""Build the design pages (self-contained, no script) from src/.

- ../index.html comes from src/index.html. It uses a small script to draw the
  sparklines and bar charts. The build renders it in headless Chromium, keeps the
  resulting static SVG and removes the script.
- ../declarations.html and ../declarations/*.html come from src/declarations/.
  Each source is a fragment (a cover and sheets). The build wraps it in
  template.html, which holds the shared style, the page list and the footer.

htmlpreview.github.io then shows the pages without running anything, which is the
`.pr/` design-doc convention.

Usage:  python3 .pr/factory-metrics/src/build.py
Needs:  chromium, chromium-browser or google-chrome on PATH.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAGES = ["index.html"]

# (source in src/declarations/, output relative to ..), nav title, group
DECLARATION_PAGES = [
    ("index.html", "declarations.html", "Overview", "Start here"),
    ("metrics.html", "declarations/metrics.html", "Metrics: how a key counts", "Start here"),
    ("count.html", "declarations/count.html", "Count (MC-01)", "Card types"),
]

browser = next(
    (b for b in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable") if shutil.which(b)),
    None,
)
if browser is None:
    sys.exit("Install Chromium or Chrome: it renders the SVG charts.")

for page in PAGES:
    src, out = HERE / page, HERE.parent / page
    dom = subprocess.run(
        [
            browser, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
            "--window-size=1600,2000", "--virtual-time-budget=4000", "--dump-dom", src.as_uri(),
        ],
        check=True, capture_output=True, text=True,
    ).stdout

    dom = re.sub(r"<script\b.*?</script>", "", dom, flags=re.S)
    assert "<script" not in dom and "<link" not in dom, f"{page}: build left a script or external stylesheet in the output"
    out.write_text("<!doctype html>\n" + dom)
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)")


def relative(target: str, current: str) -> str:
    return os.path.relpath(HERE.parent / target, (HERE.parent / current).parent)


def sheets_nav(content: str) -> str:
    sheets = re.findall(
        r'<section class="sheet" id="(s\d+)">\s*<header><span class="no">(\d+)</span><h2>(.*?)</h2>',
        content,
        flags=re.S,
    )
    return "\n".join(f'  <a href="#{i}"><b>{n}</b>{t}</a>' for i, n, t in sheets)


def pages_nav(current: str) -> str:
    lines, group = [], None
    for _, out, title, grp in DECLARATION_PAGES:
        if grp != group:
            lines.append(f'  <div class="grp">{grp}</div>' if group else "")
            group = grp
        cls = ' class="cur"' if out == current else ""
        lines.append(f'  <a{cls} href="{relative(out, current)}">{title}</a>')
    return "\n".join(line for line in lines if line)


template = (HERE / "declarations" / "template.html").read_text()
for name, out_name, title, _ in DECLARATION_PAGES:
    content = (HERE / "declarations" / name).read_text()
    root = relative("declarations.html", out_name)
    content = content.replace("{{overview}}", root)
    for _, other, _, _ in DECLARATION_PAGES:
        content = content.replace("{{to:%s}}" % other, relative(other, out_name))
    html = (
        template.replace("{{title}}", f"Factory metric cards: {title}")
        .replace("{{pages_nav}}", pages_nav(out_name))
        .replace("{{sheets_nav}}", sheets_nav(content))
        .replace("{{content}}", content)
    )
    assert "<script" not in html and "<link" not in html, f"{out_name}: script or link in output"
    assert "{{" not in html, f"{out_name}: unresolved placeholder"
    out = HERE.parent / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)")
