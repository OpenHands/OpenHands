// Import the script-automation fixture next to this file into a running
// Canvas and screenshot the preview and the result. Pass "before" to expect
// the pre-fix rejection toast instead of the preview.
//
//   npm run dev:mock            # Canvas with the MSW mock API on :3001
//   node .pr/import-script-automation/import-evidence.mjs [before|after]
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("@playwright/test");

const here = dirname(fileURLToPath(import.meta.url));
const phase = process.argv[2] ?? "after";
const origin = process.env.ORIGIN ?? "http://localhost:3001";
const fixture = join(here, "socials-monitor.automation.json");

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? "chrome" });
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
await context.addInitScript((origin) => {
  const backend = { id: "default-local", name: "Local", host: origin, apiKey: "", kind: "local" };
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-backends", JSON.stringify([backend]));
  localStorage.setItem("openhands-active-backend", JSON.stringify({ backendId: backend.id, orgId: null }));
  localStorage.setItem("analytics-consent", "false");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem("openhands-telemetry-first-use", "true");
}, origin);

const page = await context.newPage();
page.on("request", (request) => {
  if (request.url().includes("/api/automation/v1")) console.log(request.method(), request.url().replace(origin, ""));
});
await page.goto(`${origin}/automations`, { waitUntil: "networkidle" });
await page.getByTestId("confirm-telemetry-preferences").click({ timeout: 5000 }).catch(() => {});

await page.getByTestId("automations-add-automation").click();
await page.getByTestId("automations-import-automation").click();
await page.getByTestId("automations-import-file").setInputFiles(fixture);
await page.waitForTimeout(1500);

if (phase === "before") {
  await page.screenshot({ path: join(here, "before.png") });
  console.log("rejected:", (await page.locator("body").innerText()).includes("spec.prompt: expected a non-empty string"));
} else {
  await page.waitForSelector('[data-testid="import-automation-modal"][data-view="preview"]');
  await page.screenshot({ path: join(here, "after-preview.png") });
  await page.getByTestId("import-automation-confirm").click();
  await page.waitForTimeout(3000);
  await page.screenshot({ path: join(here, "after-imported.png") });
  console.log("imported:", (await page.locator("body").innerText()).includes("as disabled."));
}
await browser.close();
