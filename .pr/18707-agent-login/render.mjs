// Renders a captured terminal log as a PNG for the PR body.
//   node .pr/18707-agent-login/render.mjs <log.txt> <out.png>
// CHROMIUM_PATH points Playwright at an existing browser when its own is missing.
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const [input, output] = process.argv.slice(2);
const escape = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
const lines = readFileSync(input, "utf8").trimEnd().split("\n");
const html = lines
  .map((line) => {
    const text = escape(line);
    if (line.startsWith("###")) return `<b style="color:#7dd3fc">${text}</b>`;
    if (/-> 000|Exited|SyntaxError|ACCEPTED|167008/.test(line)) return `<span style="color:#fca5a5">${text}</span>`;
    if (/401$|400 Bad|-> 200|EPIPE|86112|"sandbox"/.test(line)) return `<span style="color:#86efac">${text}</span>`;
    return text;
  })
  .join("\n");
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const page = await browser.newPage({ viewport: { width: 1100, height: 400 }, deviceScaleFactor: 2 });
await page.setContent(
  `<pre style="margin:0;padding:20px;background:#111827;color:#e5e7eb;font:13px/1.45 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-all">${html}</pre>`,
);
await page.locator("pre").screenshot({ path: output });
await browser.close();
