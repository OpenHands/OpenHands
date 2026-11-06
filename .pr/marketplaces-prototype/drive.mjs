// Drives the Marketplaces prototype in Canvas mock mode and saves screenshots.
// Usage: node drive_marketplaces.mjs <baseUrl> <outDir>
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
  if (m.type() === "error") errors.push(`console: ${m.text().slice(0, 200)}`);
});

const dismissConsent = async () => {
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.count()) {
    await consent.click();
    await page.getByTestId("telemetry-consent-form").waitFor({ state: "detached" });
  }
};
const shot = async (name) => {
  await page.screenshot({ path: join(outDir, `${name}.png`), fullPage: false });
  console.log(`shot ${name}`);
};
const text = async (testId) =>
  (await page.getByTestId(testId).first().textContent())?.trim();

// Wait until the mock worker is up: the first load of the list page.
await page.goto(`${baseUrl}/marketplaces`, { waitUntil: "networkidle" });
await dismissConsent();
await page.getByTestId("marketplace-card-openhands-extensions").waitFor({ timeout: 60000 });
await dismissConsent();
console.log("list: cards =", await page.locator('[data-testid^="marketplace-card-"]').count());
console.log("list: default counts:", await page.locator('[data-testid^="marketplace-count-openhands-extensions-"]').allTextContents());
await shot("01-list");

// Open the default marketplace page.
await page.getByTestId("marketplace-card-openhands-extensions").click();
await page.waitForURL(/\/marketplaces\/openhands-extensions$/);
await page.getByTestId("marketplace-catalog-toolbar").waitFor();
const kindAll = await text("marketplace-kind-all");
console.log("detail: kind filter all =", kindAll);
console.log("detail: entries rendered =", await page.locator('[data-testid^="catalog-entry-kind-"]').count());
await shot("02-default-marketplace");

// Filter by kind and search.
await page.getByTestId("marketplace-kind-plugin").click();
console.log("detail: plugin cards =", await page.locator('[data-testid^="catalog-entry-plugin-"]').count());
await page.getByTestId("marketplace-kind-all").click();
await page.getByTestId("marketplace-search-input").fill("slack");
await page.waitForTimeout(200);
console.log("detail: 'slack' matches =", await page.locator('[data-testid^="catalog-entry-kind-"]').count());
await shot("03-search-slack");
await page.getByTestId("marketplace-search-input").fill("");

// Install an MCP server and a skill from the trusted default marketplace.
await page.getByTestId("marketplace-kind-mcp").click();
const firstMcpInstall = page.locator('[data-testid^="catalog-entry-install-mcp-"]').first();
const mcpId = await firstMcpInstall.getAttribute("data-testid");
await firstMcpInstall.click();
await page.locator(`[data-testid="${mcpId.replace("-install-", "-installed-")}"]`).waitFor();
console.log("detail: installed", mcpId.replace("catalog-entry-install-", ""), "->", await page.locator(`[data-testid="${mcpId.replace("-install-", "-installed-")}"]`).textContent());
await page.getByTestId("marketplace-kind-skill").click();
const firstSkillInstall = page.locator('[data-testid^="catalog-entry-install-skill-"]').first();
const skillId = await firstSkillInstall.getAttribute("data-testid");
await firstSkillInstall.click();
await page.locator(`[data-testid="${skillId.replace("-install-", "-installed-")}"]`).waitFor();
console.log("detail: installed", skillId.replace("catalog-entry-install-", ""));
await page.getByTestId("marketplace-install-filter-installed").click();
console.log("detail: installed filter shows", await page.locator('[data-testid^="catalog-entry-kind-"]').count(), "entries");
await shot("04-installed-filter");

// Refresh the marketplace: tracked installs gain an update.
await page.getByTestId("marketplace-detail-refresh").click();
await page.locator('[data-testid^="catalog-entry-update-"]').first().waitFor({ timeout: 10000 });
console.log("detail: update buttons after refresh =", await page.locator('[data-testid^="catalog-entry-update-"]').count());
await shot("05-update-available");
await page.locator('[data-testid^="catalog-entry-update-"]').first().click();
await page.waitForTimeout(300);
console.log("detail: update buttons after one update =", await page.locator('[data-testid^="catalog-entry-update-"]').count());

// Untrusted marketplace: runs-code entries are blocked until trusted.
await page.goto(`${baseUrl}/marketplaces/acme-internal`, { waitUntil: "networkidle" });
await dismissConsent();
await page.getByTestId("marketplace-catalog-toolbar").waitFor();
console.log("acme: errors panel =", (await text("marketplace-catalog-errors"))?.slice(0, 80));
console.log("acme: also-in badges =", await page.locator('[data-testid^="catalog-entry-also-in-"]').allTextContents());
console.log("acme: acme-deploy install disabled =", await page.getByTestId("catalog-entry-install-plugin-acme-deploy").isDisabled());
console.log("acme: acme-oncall install disabled =", await page.getByTestId("catalog-entry-install-skill-acme-oncall").isDisabled());
await shot("06-acme-untrusted");
await page.getByTestId("marketplace-trust-panel").getByRole("switch").click();
await page.waitForTimeout(400);
console.log("acme: acme-deploy install disabled after trust =", await page.getByTestId("catalog-entry-install-plugin-acme-deploy").isDisabled());
await page.getByTestId("catalog-entry-install-plugin-acme-deploy").click();
await page.getByTestId("catalog-entry-installed-plugin-acme-deploy").waitFor();
console.log("acme: acme-deploy installed; toggle present =", await page.getByTestId("catalog-entry-toggle-plugin-acme-deploy").count());
await shot("07-acme-trusted-installed");

// Add a marketplace from the list page.
await page.goto(`${baseUrl}/marketplaces`, { waitUntil: "networkidle" });
await dismissConsent();
await page.getByTestId("marketplaces-add-button").click();
await page.getByTestId("add-marketplace-source-input").fill("github:obra/superpowers");
await page.getByTestId("add-marketplace-modal").getByRole("switch").click();
await shot("08-add-form");
await page.getByTestId("add-marketplace-submit").click();
await page.waitForURL(/\/marketplaces\/superpowers$/, { timeout: 10000 });
await page.getByTestId("marketplace-catalog-toolbar").waitFor();
console.log("added: superpowers kind filter all =", await text("marketplace-kind-all"));
await shot("09-superpowers");

// Add fails for a source that cannot be parsed, and for a private repo without a token.
await page.goto(`${baseUrl}/marketplaces`, { waitUntil: "networkidle" });
await dismissConsent();
await page.getByTestId("marketplaces-add-button").click();
await page.getByTestId("add-marketplace-source-input").fill("github:acme-corp/private-extensions");
await page.getByTestId("add-marketplace-submit").click();
await page.waitForTimeout(600);
console.log("add: private without token keeps the form open =", await page.getByTestId("add-marketplace-modal").count());
await shot("10-add-private-error");
await page.getByTestId("add-marketplace-modal-close").click();

// Remove the added marketplace.
console.log("list: cards before remove =", await page.locator('[data-testid^="marketplace-card-"]').count());
await page.getByTestId("marketplace-remove-superpowers").click();
await page.getByTestId("confirm-button").click();
await page.waitForTimeout(500);
console.log("list: cards after remove =", await page.locator('[data-testid^="marketplace-card-"]').count());
await shot("11-list-after-remove");

// Reload keeps the state (sessionStorage).
await page.reload({ waitUntil: "networkidle" });
await dismissConsent();
await page.getByTestId("marketplace-card-openhands-extensions").waitFor();
console.log("list: acme trusted after reload =", await text("marketplace-trust-acme-internal"));

// Sidebar entry
console.log("nav: Marketplaces link present =", await page.locator('a[href="/marketplaces"]').count());

console.log("errors:", errors.length ? errors : "none");
await browser.close();
