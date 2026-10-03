import { chromium } from "@playwright/test";

const baseUrl = "http://localhost:3001";
const rca = {
  summary: "Connection pool exhaustion under load",
  evidence: ["pg pool max=5 reached", "502s spiked at 14:03 UTC"],
  suspected_components: ["db-pool"],
  suspected_files: ["src/db/pool.ts"],
  recommended_action: "Raise the pool limit and add backpressure",
  source: "holmesgpt",
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(baseUrl, { waitUntil: "networkidle" });

await page.getByTestId("onboarding-backend-kind-option-local").click();
await page.getByTestId("onboarding-backend-next").click();
await page.waitForTimeout(3000);

// Dismiss telemetry consent (PATCH persists the choice in MSW).
const consentBtn = page.getByTestId("confirm-telemetry-preferences");
if (await consentBtn.isVisible().catch(() => false)) {
  await consentBtn.click();
  await page.waitForTimeout(3000);
}

await page.getByTestId("onboarding-agent-next").click();
await page.waitForTimeout(1500);
await page.getByTestId("onboarding-llm-next").click();
await page.waitForTimeout(1500);
const skip = page.getByTestId("onboarding-skip");
if (await skip.isVisible().catch(() => false)) await skip.click();
await page.waitForTimeout(1500);
await page.screenshot({ path: ".pr/debug-landing8.png" });

await page.getByTestId("open-rca-import").waitFor({ state: "visible" });
await page.waitForTimeout(500);

await page.getByTestId("open-rca-import").click();
await page.getByTestId("rca-import-modal").waitFor();
await page.getByTestId("rca-json-input").fill(JSON.stringify(rca, null, 2));
await page.screenshot({ path: ".pr/rca-import-modal.png" });

await page.getByTestId("rca-import-apply").click();
await page.getByTestId("rca-attached-chip").waitFor();
await page.screenshot({ path: ".pr/rca-attached-chip.png" });

await browser.close();
console.log("saved screenshots");
