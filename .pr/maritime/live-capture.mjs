// Live Agent Canvas capture against a real Maritime account.
//
//   MARITIME_WEB=1 MARITIME_API_KEY=mk_... npm run dev:static -- --port 12000
//   node .pr/maritime/live-capture.mjs <out-dir> create   # create, open, connect
//   node .pr/maritime/live-capture.mjs <out-dir> manage   # sleep, delete
//
// The Canvas server holds the API key (MARITIME_API_KEY at startup); the page
// never sees or types it. It creates one agent and deletes it again.
import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs";

const baseUrl = process.env.CANVAS_URL ?? "http://localhost:12000";
const outDir = process.argv[2];
const step = process.argv[3] ?? "all";
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({
  viewport: { width: 1280, height: 860 },
  deviceScaleFactor: 2,
  colorScheme: "dark",
  storageState: fs.existsSync(path.join(outDir, "state.json"))
    ? path.join(outDir, "state.json")
    : undefined,
});
const page = await context.newPage();
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
const failed = [];
page.on("response", (r) => {
  if (r.status() >= 400) failed.push(`${r.status()} ${r.request().method()} ${r.url().replace(/session_api_key=[^&]+/, "session_api_key=***")}`);
});
await page.addInitScript(() => localStorage.setItem("openhands-onboarded", "true"));

const log = (m) => console.log(`[${new Date().toISOString().slice(11, 19)}] ${m}`);
async function shot(name) {
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(outDir, `${name}.png`) });
  log(`saved ${name}.png`);
}

// The first visit to each backend asks about usage data: decline it.
async function declineUsageData() {
  const consent = page.getByRole("button", { name: "Confirm preferences" });
  if (await consent.isVisible().catch(() => false)) {
    const box = page.getByRole("checkbox", { name: /usage data/i });
    if (await box.isChecked().catch(() => false)) await box.uncheck();
    await consent.click();
    await page.waitForTimeout(1500);
  }
}

async function openMaritime() {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  await declineUsageData();
  const skip = page.getByRole("button", { name: "Skip for now" });
  if (await skip.isVisible().catch(() => false)) await skip.click();
  const chooser = page.getByTestId("add-backend-chooser");
  if (!(await chooser.isVisible().catch(() => false))) {
    const add = page.getByRole("button", { name: "Add Backend" }).first();
    if (await add.isVisible().catch(() => false)) {
      await add.click();
    } else {
      await page.getByTestId("backend-selector").first().click();
      for (const id of ["add-backend-menu-item", "manage-backends-add"]) {
        const el = page.getByTestId(id).first();
        if (await el.isVisible().catch(() => false)) await el.click();
      }
    }
  }
  await chooser.waitFor({ timeout: 20_000 });
  await page.getByTestId("add-backend-option-providers").click();
}

try {
  if (step === "all" || step === "create") {
    await openMaritime();
    await shot("01-live-providers");
    await page.getByTestId("add-backend-provider-maritime").click();
    await page
      .locator('[data-testid="maritime-agents"], [data-testid="maritime-no-agents"]')
      .first()
      .waitFor({ timeout: 30_000 });
    await page.getByTestId("maritime-account").waitFor({ timeout: 30_000 });
    await shot("02-live-signed-in");

    await page.getByRole("button", { name: "New agent" }).click();
    await page.getByTestId("maritime-new-agent-form").waitFor();
    await page.getByLabel("Name").fill("openhands-live-demo");
    await shot("03-live-new-agent");
    await page.getByRole("button", { name: "Create and open" }).click();
    await page.waitForTimeout(8000);
    await shot("04-live-creating");

    // Canvas opens the agent when its agent server answers: the modal closes.
    log("waiting for the agent to start and Canvas to connect (up to 10 min)");
    await page
      .getByTestId("maritime-connect-panel")
      .waitFor({ state: "detached", timeout: 600_000 });
    await page.waitForLoadState("networkidle").catch(() => {});
    await page.waitForTimeout(4000);
    await declineUsageData();
    await shot("05-live-connected");
    log(`url after connect: ${page.url()}`);
    await context.storageState({ path: path.join(outDir, "state.json") });
  }

  if (step === "connected") {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.waitForTimeout(3000);
    const consent = page.getByRole("button", { name: "Confirm preferences" });
    if (await consent.isVisible().catch(() => false)) {
      const box = page.getByRole("checkbox", { name: /usage data/i });
      if (await box.isChecked().catch(() => false)) await box.uncheck();
      await consent.click();
      await page.waitForTimeout(1500);
    }
    await shot("05-live-connected");
    await context.storageState({ path: path.join(outDir, "state.json") });
  }

  if (step === "all" || step === "manage") {
    await openMaritime();
    await page.getByTestId("add-backend-provider-maritime").click();
    await page.getByTestId("maritime-agents").waitFor({ timeout: 30_000 });
    await shot("06-live-agent-running");

    const sleep = page.locator('[data-testid^="maritime-agent-sleep-"]').first();
    await sleep.click();
    await page.getByText("Sleeping").first().waitFor({ timeout: 120_000 });
    await shot("07-live-agent-sleeping");

    await page.locator('[data-testid^="maritime-agent-delete-"]').first().click();
    await page.waitForTimeout(800);
    await shot("08-live-delete-confirmation");
    await page.getByRole("button", { name: /^delete/i }).last().click();
    await page
      .locator('[data-testid^="maritime-agent-delete-"]')
      .first()
      .waitFor({ state: "detached", timeout: 120_000 })
      .catch(() => {});
    await shot("09-live-after-delete");
  }
} catch (error) {
  log(`FAILED: ${error.message.split("\n")[0]}`);
  await page.screenshot({ path: path.join(outDir, `fail-${step}.png`) });
  process.exitCode = 1;
} finally {
  if (failed.length) log(`failed requests:\n  ${[...new Set(failed)].slice(0, 15).join("\n  ")}`);
  if (errors.length) log(`console errors (${errors.length}): ${errors.slice(0, 5).join(" | ").slice(0, 600)}`);
  await browser.close();
}
