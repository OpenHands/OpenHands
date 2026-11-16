// Evidence capture for #19755: the document shell opts out of browser
// translation (translate="no" + notranslate meta), so translation engines
// cannot rewrite React-managed text nodes.
// Requires the MSW dev server: VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3193
// Usage: node .pr/bugs/browser-translate-guard/capture.mjs before|after <outputDir>
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const mode = process.argv[2] ?? "after";
const output = process.argv[3] ?? `.pr/bugs/browser-translate-guard/${mode}`;
await mkdir(output, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 960 } });
page.setDefaultTimeout(20000);

await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "default-local",
        name: "Mock backend",
        host: location.origin,
        apiKey: "mock-session-key",
        kind: "local",
      },
    ]),
  );
  localStorage.setItem(
    "openhands-active-backend",
    JSON.stringify({ backendId: "default-local", orgId: null }),
  );
  localStorage.setItem("analytics-consent", "false");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem("openhands-telemetry-first-use", "true");
});

try {
  await page.goto("http://127.0.0.1:3193/", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(1500);

  const shell = await page.evaluate(() => ({
    htmlTranslate: document.documentElement.translate,
    googleMeta: document
      .querySelector('meta[name="google"]')
      ?.getAttribute("content"),
  }));
  console.log(JSON.stringify({ mode, shell }, null, 2));

  await page.screenshot({ path: `${output}/01-app-shell.png` });
} finally {
  await browser.close();
}
