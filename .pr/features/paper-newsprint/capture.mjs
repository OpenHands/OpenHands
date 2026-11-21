// Evidence capture for #19596: Paper as grey newsprint with carbon ink.
// Requires the MSW dev server: VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3193
// Usage: node .pr/features/paper-newsprint/capture.mjs before|after <outputDir>
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const mode = process.argv[2] ?? "before";
const output = process.argv[3] ?? `.pr/features/paper-newsprint/${mode}`;
await mkdir(output, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 960 } });
page.setDefaultTimeout(20000);

await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-color-theme", "paper");
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
  // 1. Templates grid: cards, pills, primary buttons, beta tags.
  await page.goto("http://127.0.0.1:3193/automations/templates", {
    waitUntil: "domcontentloaded",
  });
  const consent = page.getByRole("button", {
    name: /Confirm preferences|确认偏好设置/,
  });
  if (await consent.isVisible().catch(() => false)) {
    const checkbox = page.getByRole("checkbox").first();
    if (await checkbox.isVisible().catch(() => false)) {
      await checkbox.uncheck().catch(() => {});
    }
    await consent.click();
    await page.waitForTimeout(500);
  }
  await page
    .getByTestId("recommended-automation-card-custom-automation")
    .waitFor();
  await page.waitForTimeout(800);

  // The consent dialog can appear over the loaded grid; dismiss it before
  // capturing.
  const consent2 = page.getByRole("button", {
    name: /Confirm preferences|确认偏好设置/,
  });
  if (await consent2.isVisible().catch(() => false)) {
    const checkbox = page.getByRole("checkbox").first();
    if (await checkbox.isVisible().catch(() => false)) {
      await checkbox.uncheck().catch(() => {});
    }
    await consent2.click({ force: true });
    await page.waitForTimeout(600);
  }

  const shell = await page.evaluate(() => ({
    pageBackground: getComputedStyle(document.body).backgroundColor,
    theme: document.documentElement.getAttribute("data-color-theme"),
  }));
  console.log(JSON.stringify({ mode, shell }, null, 2));
  await page.screenshot({ path: `${output}/01-templates-paper.png` });

  // 2. Phone width: sidebar drawer + theme via settings at phone viewport.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("http://127.0.0.1:3193/settings/app", {
    waitUntil: "domcontentloaded",
  });
  await page
    .getByTestId("color-theme-input")
    .waitFor({ timeout: 20000 })
    .catch(() => {});
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${output}/02-phone-theme-settings.png` });
} finally {
  await browser.close();
}
