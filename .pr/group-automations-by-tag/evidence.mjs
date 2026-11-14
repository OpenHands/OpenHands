// Captures the Automations dashboard grouped by tag, against the mock API.
//
//   npm run dev:mock -- --port 3055
//   node .pr/group-automations-by-tag/evidence.mjs http://localhost:3055
//
// Writes the screenshots next to this file and prints the sections it sees.
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const baseUrl = process.argv[2] ?? "http://localhost:3055";
const outDir = dirname(fileURLToPath(import.meta.url));
const shot = (name) => ({ path: join(outDir, name) });

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1280, height: 900 },
});
await context.addInitScript(() => {
  window.localStorage.setItem("analytics-consent", "false");
  window.localStorage.setItem("openhands-telemetry-consent", "denied");
  window.localStorage.setItem("openhands-telemetry-first-use", "true");
  window.localStorage.setItem("openhands-onboarded", "1");
  window.localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "default-local",
        name: "Local",
        host: window.location.origin,
        apiKey: "",
        kind: "local",
      },
    ]),
  );
  window.localStorage.setItem(
    "openhands-active-backend",
    JSON.stringify({ backendId: "default-local", orgId: null }),
  );
});
const page = await context.newPage();

async function openDashboard() {
  await page.goto(`${baseUrl}/automations`);
  // Mock settings carry no analytics choice, so the consent prompt covers the
  // page until it is answered.
  await page
    .getByTestId("confirm-telemetry-preferences")
    .click({ timeout: 8000 })
    .catch(() => {});
  await page.locator("section h2").first().waitFor();
}

// Each section as "title (badge count, automations listed)".
const readSections = () =>
  page.evaluate(() =>
    Array.from(document.querySelectorAll("section"))
      .filter((section) => section.querySelector(":scope > div > h2"))
      .map((section) => {
        const title = section.querySelector(":scope > div > h2").textContent;
        const badge = section.querySelector(":scope > div > span").textContent;
        const listed = Array.from(
          section.querySelectorAll("[data-testid]"),
        ).filter((element) =>
          /^automation-(card|list-row)-[0-9a-f-]{36}$/.test(
            element.getAttribute("data-testid"),
          ),
        ).length;
        return `${title} (${badge}, ${listed} listed)`;
      })
      .join(" | "),
  );

const groupByTrigger = page
  .getByTestId("automations-group-by")
  .getByTestId("dropdown-trigger");

await openDashboard();
console.log("default sections:    ", await readSections());

await groupByTrigger.click();
await page.screenshot(shot("01-group-by-menu.png"));
await page.getByTestId("automations-group-by-tag").click();
console.log("grouped by tag:      ", await readSections());
await page.screenshot(shot("02-grouped-by-tag-grid.png"));

await openDashboard();
await page.locator('section h2:text-is("No tags")').waitFor();
console.log("after a reload:      ", await readSections());

await page.getByTestId("automations-view-toggle").click();
await page.getByTestId("automations-view-toggle-list").click();
console.log("list view:           ", await readSections());
await page.screenshot(shot("03-grouped-by-tag-list.png"));

await page.setViewportSize({ width: 390, height: 800 });
await page.screenshot(shot("04-phone-width.png"));
await page.setViewportSize({ width: 1280, height: 900 });

await groupByTrigger.click();
await page.getByTestId("automations-group-by-status").click();
console.log("back to status:      ", await readSections());

await browser.close();
