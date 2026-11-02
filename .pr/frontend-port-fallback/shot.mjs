// Usage:
//   node shot.mjs url <url> <out.png> <playwright-index.mjs>
//   node shot.mjs log <log-file> <out.png> <playwright-index.mjs> <title>
import { readFileSync } from "node:fs";

const [mode, input, out, playwrightPath, title = ""] = process.argv.slice(2);
const { chromium } = await import(playwrightPath);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

if (mode === "url") {
  await page.goto(input, { waitUntil: "networkidle", timeout: 60_000 });
  await page.waitForTimeout(3000);
} else {
  // Strip ANSI colour codes from the captured launcher output.
  const text = readFileSync(input, "utf8").replace(/\x1b\[[0-9;]*m/g, "");
  const escape = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  await page.setContent(
    `<body style="margin:0;background:#1e1e1e;color:#ddd;font:14px/1.45 Menlo,monospace">` +
      `<div style="background:#333;color:#fff;padding:8px 16px;font:13px system-ui">${escape(title)}</div>` +
      `<pre style="margin:0;padding:16px;white-space:pre-wrap">${escape(text)}</pre></body>`,
  );
  await page.setViewportSize({
    width: 1280,
    height: Math.min(
      1400,
      await page.evaluate(() => document.body.scrollHeight),
    ),
  });
}

await page.screenshot({ path: out, fullPage: false });
await browser.close();
console.log(`saved ${out}`);
