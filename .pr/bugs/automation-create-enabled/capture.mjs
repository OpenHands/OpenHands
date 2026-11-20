// Evidence capture for #19718: completing the automation setup wizard creates
// the automation enabled. The visible proof is the create request body the
// wizard sends (the MSW echo always reports enabled, so the request body is
// the behavioral difference).
// Requires the MSW dev server: VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3193
// Usage: node .pr/bugs/automation-create-enabled/capture.mjs before|after <outputDir>
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const mode = process.argv[2] ?? "before";
const output = process.argv[3] ?? `.pr/bugs/automation-create-enabled/${mode}`;
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
  const createBodies = [];
  page.on("request", (request) => {
    const url = request.url();
    if (
      request.method() === "POST" &&
      /\/api\/automation\/v1\/(preset\/|drafts|validate|$)/.test(
        new URL(url).pathname + "/",
      )
    ) {
      createBodies.push(`${new URL(url).pathname} ${request.postData()}`);
    }
  });

  await page.goto("http://127.0.0.1:3193/automations/setup", {
    waitUntil: "domcontentloaded",
  });
  await page.getByTestId("automation-setup-name").waitFor();
  await page.waitForTimeout(2000);

  // Dismiss the analytics consent dialog when it covers the wizard.
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
  await page.waitForTimeout(300);

  const name = page.getByTestId("automation-setup-name");
  if ((await name.count()) > 0 && (await name.getAttribute("value")) === "") {
    await name.fill("Daily greeting");
  }
  await page
    .getByTestId("automation-setup-prompt")
    .fill("Reply with only the word OK. Do not run any tools.");
  await page.screenshot({ path: `${output}/01-wizard-filled.png` });

  // The wizard renders inside a modal; its backdrop sits behind the dialog
  // and intercepts Playwright's actionability point, so click the visible,
  // enabled button directly.
  await page.getByTestId("automation-setup-create").click({ force: true });
  await page.waitForURL(/\/automations\//, { timeout: 15000 });
  await page.waitForTimeout(1200);

  console.log(
    JSON.stringify({ mode, createRequestBodies: createBodies }, null, 2),
  );
  await page.screenshot({ path: `${output}/02-after-create.png` });
} finally {
  await browser.close();
}
