// Evidence capture for #17959 (problem 2): optional integrations counted as
// "MCPs to connect before launch".
// Requires the MSW dev server: VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3193
// Usage: node .pr/bugs/automation-template-counts/capture.mjs before|after <outputDir>
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const mode = process.argv[2] ?? "before";
const output = process.argv[3] ?? `.pr/bugs/automation-template-counts/${mode}`;
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
  await page.goto("http://127.0.0.1:3193/automations/templates", {
    waitUntil: "domcontentloaded",
  });
  const card = page.getByTestId("recommended-automation-card-custom-automation");
  await card.waitFor();
  await page.waitForTimeout(800);

  const pills = page.getByTestId("recommended-automation-pills-custom-automation");
  const pillText = await pills.innerText();
  console.log(JSON.stringify({ mode, pillText }, null, 2));

  await card.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await card.screenshot({ path: `${output}/01-custom-automation-card.png` });
  await page.screenshot({ path: `${output}/02-templates-page.png` });
} finally {
  await browser.close();
}
