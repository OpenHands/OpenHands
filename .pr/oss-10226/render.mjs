// Renders a vitest log as a PNG for the PR evidence.
// Usage: node .pr/oss-10226/render.mjs <log> <out.png> <title>
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const [log, out, title] = process.argv.slice(2);
const lines = readFileSync(log, "utf8")
  .replace(/\x1b\[[0-9;]*m/g, "")
  .split("\n")
  .filter((l) => /[✓×]|Test Files|Tests {2}/.test(l));
const escape = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const body = lines
  .map((l) => {
    let color = "#d4d4d4";
    if (l.includes("×") || /failed/.test(l)) color = "#f87171";
    else if (l.includes("✓") || /passed/.test(l)) color = "#4ade80";
    return `<div style="color:${color}">${escape(l)}</div>`;
  })
  .join("");

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 200 } });
await page.setContent(
  `<body style="margin:0;background:#1e1e1e;font:13px/1.5 Menlo,monospace;padding:16px;white-space:pre">` +
    `<div style="color:#fff;font-weight:bold;margin-bottom:8px">${escape(title)}</div>${body}</body>`,
);
await page.screenshot({ path: out, fullPage: true });
await browser.close();
