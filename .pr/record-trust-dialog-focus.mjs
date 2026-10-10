// Capture the actual local Agent Canvas behavior on a doctored control-openhands
// run. Usage: node .pr/record-trust-dialog-focus.mjs <before|after|after-phone> <base-url>
// The demo Canvas app must already be installed but disabled in that run.
import { mkdir, rename } from "node:fs/promises";
import { chromium } from "playwright";

const [mode, baseUrl] = process.argv.slice(2);
if (
  !["before", "after", "after-phone"].includes(mode) ||
  !/^http:\/\/127\.0\.0\.1:\d+$/.test(baseUrl)
) {
  throw new Error(
    "Usage: <before|after|after-phone> http://127.0.0.1:<run-port>",
  );
}

const outputDir = new URL("./trust-dialog-focus/", import.meta.url);
await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const viewport =
  mode === "after-phone"
    ? { width: 390, height: 844 }
    : { width: 1440, height: 1000 };
const context = await browser.newContext({
  viewport,
  recordVideo: { dir: outputDir.pathname, size: viewport },
});
await context.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem(
    "openhands-telemetry-consent-pending-cloud-sync",
    "denied",
  );
});
const page = await context.newPage();
await page.goto(`${baseUrl}/apps`);
const appSwitch = page
  .getByTestId("canvas-extension-card-demo-page")
  .getByRole("switch");
await appSwitch.waitFor();
await appSwitch.focus();
await page.keyboard.press("Space");
const dialog = page.getByTestId("confirmation-modal");
await dialog.waitFor();
await page.waitForTimeout(800);
await page.keyboard.press("Tab");
await page.waitForTimeout(800);
await page.keyboard.press("Enter");
await page.waitForTimeout(1800);

const result = {
  mode,
  dialogOpen: await dialog.isVisible(),
  activeTestId: await page.evaluate(() =>
    document.activeElement?.getAttribute("data-testid"),
  ),
  toastText: await page.locator('[role="status"]').allTextContents(),
};
const video = page.video();
await context.close();
await browser.close();
if (!video) throw new Error("Playwright did not record a video");
await rename(
  await video.path(),
  new URL(`./trust-dialog-focus/${mode}.webm`, import.meta.url),
);
console.log(JSON.stringify(result));
