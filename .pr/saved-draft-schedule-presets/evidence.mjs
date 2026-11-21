import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const outDir = dirname(fileURLToPath(import.meta.url));
const mode = process.argv[2] ?? "after";
const baseUrl = process.argv[3] ?? "http://127.0.0.1:3055";

async function launchBrowser() {
  try {
    return await chromium.launch({ headless: true });
  } catch {
    return await chromium.launch({ channel: "msedge", headless: true });
  }
}

async function readScheduleState(page) {
  return page.evaluate(() => {
    const frequencies = [
      "once",
      "hourly",
      "daily",
      "weekdays",
      "weekly",
      "custom",
    ];
    const selectedFrequency =
      frequencies.find(
        (f) =>
          document
            .querySelector(`[data-testid="automation-setup-frequency-${f}"]`)
            ?.getAttribute("aria-checked") === "true",
      ) ?? null;
    const weekday =
      document.querySelector('[data-testid="automation-setup-weekday"]')
        ?.value ?? null;
    const time =
      document.querySelector('[data-testid="automation-setup-time"]')?.value ??
      null;
    const customSchedule =
      document.querySelector('[data-testid="automation-setup-custom-schedule"]')
        ?.value ?? null;
    const saveState =
      document
        .querySelector('[data-testid="automation-setup-save-state"]')
        ?.textContent?.trim() ?? null;
    const testEnabled = !document
      .querySelector('[data-testid="automation-setup-test"]')
      ?.hasAttribute("disabled");
    return {
      selectedFrequency,
      weekday,
      time,
      customSchedule,
      saveState,
      testEnabled,
    };
  });
}

await mkdir(outDir, { recursive: true });

const browser = await launchBrowser();
try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 920 },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(30_000);
  await page.addInitScript(() => {
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

  await page.goto(`${baseUrl}/automations`);
  await page.getByTestId("confirm-telemetry-preferences").click();
  await page
    .getByTestId("telemetry-consent-form")
    .waitFor({ state: "detached" });

  await page.getByTestId("automations-add-automation").click();
  await page.getByTestId("automations-add-automation-create").click();
  await page.waitForURL(/\/automations\/setup$/);

  await page.getByTestId("automation-setup-frequency-weekly").click();
  await page.getByTestId("automation-setup-weekday").selectOption("3");
  await page.getByTestId("automation-setup-time").fill("10:30");
  await page.getByTestId("automation-setup-name").fill("Weekly PR Digest");
  await page
    .getByTestId("automation-setup-prompt")
    .fill("Summarize merged pull requests from the past week.");
  await page.getByTestId("automation-setup-save-draft").click();

  await page.waitForURL(/draftId=/);
  await page.getByTestId("automation-setup-draft-details").waitFor();
  await page.mouse.move(10, 10);
  await page.evaluate(() => {
    for (const el of document.querySelectorAll("*")) {
      if (el.scrollTop > 0) el.scrollTop = 0;
    }
  });
  await page.waitForTimeout(4500);

  const savedState = await readScheduleState(page);
  console.log(`${mode} (when saved):`, JSON.stringify(savedState));
  if (mode === "before") {
    await page.screenshot({
      path: resolve(outDir, "01-saved-weekly-draft.png"),
    });
  }

  await page.getByTestId("automation-page-back").click();
  await page.waitForURL(/\/automations(\?|$)/);
  const resumeBtn = page
    .locator('[data-testid^="automation-setup-draft-resume-"]')
    .first();
  await resumeBtn.waitFor();
  await resumeBtn.click();
  await page.waitForURL(/\/automations\/setup\?draftId=/);
  await page.getByTestId("automation-setup-draft-details").waitFor();
  await page.waitForTimeout(400);

  const resumedState = await readScheduleState(page);
  console.log(`${mode} (after resume):`, JSON.stringify(resumedState));
  await page.screenshot({
    path: resolve(
      outDir,
      mode === "before"
        ? "02-main-resumed-becomes-custom.png"
        : "03-pr-resumed-keeps-weekly.png",
    ),
  });
} finally {
  await browser.close();
}
