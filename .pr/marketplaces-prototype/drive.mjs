// Drives the Marketplaces prototype (single-page layout) in Canvas mock mode.
// Usage: node drive_v2.mjs <baseUrl> <outDir>
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

const [baseUrl, outDir] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
await page.addInitScript(() => {
  const backend = { id: "mock-local", name: "Mock local", host: location.origin, apiKey: "marketplaces-dev", kind: "local" };
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-backends", JSON.stringify([backend]));
  const selection = JSON.stringify({ backendId: backend.id, orgId: null });
  localStorage.setItem("openhands-active-backend", selection);
  sessionStorage.setItem("openhands-active-backend", selection);
  localStorage.setItem("openhands-sidebar-onboarding-checklist-dismissed", "true");
});
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error" && !m.text().includes("502")) errors.push(`console: ${m.text().slice(0, 200)}`);
});
const dismissConsent = async () => {
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.count()) {
    await consent.click();
    await page.getByTestId("telemetry-consent-form").waitFor({ state: "detached" });
  }
};
const shot = async (name) => {
  await page.screenshot({ path: join(outDir, `${name}.png`) });
  console.log(`shot ${name}`);
};
const text = async (testId) => (await page.getByTestId(testId).first().textContent())?.trim();
const cards = () => page.locator('[data-testid^="catalog-entry-kind-"]').count();
const pickerInput = () => page.getByLabel("Marketplace", { exact: true });
const pick = async (name) => {
  await pickerInput().click();
  await page.getByRole("option", { name: new RegExp(`^${name} `) }).click();
  await page.waitForURL(new RegExp(`/marketplaces/${name}$`));
  await page.getByTestId("marketplace-kind-tabs").waitFor();
};

await page.goto(`${baseUrl}/marketplaces`, { waitUntil: "networkidle" });
await dismissConsent();
await page.getByTestId("marketplace-kind-tabs").waitFor({ timeout: 60000 });
console.log("no customize rail on this page =", (await page.getByTestId("extensions-navbar-desktop").count()) === 0);
console.log("default: tabs =", await page.locator('[data-testid^="marketplace-tab-"]').allTextContents());
console.log("default: results =", await text("marketplace-result-summary"));
await shot("v2-01-default");

// Tabs
await page.getByTestId("marketplace-tab-skill").click();
console.log("skills tab: cards =", await cards(), "results =", await text("marketplace-result-summary"));
await page.getByTestId("marketplace-tab-automation").click();
console.log("automation tab: cards =", await cards());
await shot("v2-02-automation-tab");

// Facets: tag + status
await page.getByTestId("marketplace-tab-all").click();
await page.locator('[data-testid^="marketplace-facet-tag-"]').first().click();
const firstTag = await page.locator('[data-testid^="marketplace-facet-tag-"]').first().getAttribute("data-testid");
console.log("facet", firstTag, "-> results =", await text("marketplace-result-summary"), "clear link =", await page.getByTestId("marketplace-clear-filters").count());
await page.getByTestId("marketplace-clear-filters").click();
console.log("cleared -> results =", await text("marketplace-result-summary"));

// Install from the trusted default, then status facet + refresh/update
await page.getByTestId("marketplace-tab-mcp").click();
await page.locator('[data-testid^="catalog-entry-install-mcp-"]').first().click();
await page.locator('[data-testid^="catalog-entry-toggle-mcp-"]').first().waitFor();
await page.getByTestId("marketplace-tab-all").click();
await page.getByTestId("marketplace-facet-status-installed").click();
console.log("status=installed -> results =", await text("marketplace-result-summary"));
await page.getByTestId("marketplace-refresh").click();
await page.locator('[data-testid^="catalog-entry-update-"]').first().waitFor({ timeout: 10000 });
console.log("after refresh: update buttons =", await page.locator('[data-testid^="catalog-entry-update-"]').count());
await shot("v2-03-installed-update");
await page.locator('[data-testid^="catalog-entry-update-"]').first().click();
await page.waitForTimeout(300);
console.log("after update: update buttons =", await page.locator('[data-testid^="catalog-entry-update-"]').count());

// Switch marketplace with the picker
await pick("acme-internal");
await page.getByTestId("marketplace-remove").waitFor();
console.log("acme: remove button =", await page.getByTestId("marketplace-remove").count(), "errors =", (await text("marketplace-catalog-errors"))?.slice(0, 40));
console.log("acme: acme-deploy disabled =", await page.getByTestId("catalog-entry-install-plugin-acme-deploy").isDisabled());
await shot("v2-04-acme-untrusted");
await page.getByTestId("marketplace-trust-control").getByRole("switch").click();
await page.waitForTimeout(400);
console.log("acme: acme-deploy disabled after trust =", await page.getByTestId("catalog-entry-install-plugin-acme-deploy").isDisabled());
await page.getByTestId("catalog-entry-install-plugin-acme-deploy").click();
await page.getByTestId("catalog-entry-toggle-plugin-acme-deploy").waitFor();
await shot("v2-05-acme-trusted");

// Add a marketplace from the header; it becomes the selected one
await page.getByTestId("marketplaces-add-button").click();
await page.getByTestId("add-marketplace-source-input").fill("github:obra/superpowers");
await page.getByTestId("add-marketplace-modal").getByRole("switch").click();
await shot("v2-06-add-form");
await page.getByTestId("add-marketplace-submit").click();
await page.waitForURL(/\/marketplaces\/superpowers$/, { timeout: 10000 });
await page.getByTestId("marketplace-kind-tabs").waitFor();
await page.waitForFunction(() => document.querySelector('[data-testid="marketplace-picker-bar"]')?.textContent?.includes("obra/superpowers"));
console.log("superpowers: picker shows =", await pickerInput().inputValue(), "tabs =", await text("marketplace-tab-all"));
await shot("v2-07-superpowers");

// Remove it from the picker bar; back to the default
await page.getByTestId("marketplace-remove").click();
await page.getByTestId("confirm-button").click();
await page.waitForURL(/\/marketplaces$/);
await page.getByTestId("marketplace-kind-tabs").waitFor();
await page.waitForFunction(() => document.querySelector('input[aria-label="Marketplace"]')?.value?.startsWith("openhands-extensions"));
console.log("after remove: picker shows =", await pickerInput().inputValue());
await pickerInput().click();
console.log("picker options =", await page.getByRole("option").allTextContents());
await shot("v2-08-picker-open");

console.log("errors:", errors.length ? errors.slice(0, 5) : "none");
await browser.close();
