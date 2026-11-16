// Evidence capture for #19363: the phone automation setup form keeps unsaved
// edits across the Form ↔ Conversation round trip.
// Requires the MSW dev server: VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3193
// Usage: node .pr/bugs/phone-form-reset/capture.mjs before|after <outputDir>
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const mode = process.argv[2] ?? "before";
const output = process.argv[3] ?? `.pr/bugs/phone-form-reset/${mode}`;
await mkdir(output, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
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
  await page.goto("http://127.0.0.1:3193/automations/setup", {
    waitUntil: "domcontentloaded",
  });
  await page.getByTestId("automation-setup-name").waitFor();
  await page.waitForTimeout(1500);

  // Dismiss the analytics consent dialog when it covers the wizard.
  const consent = page.getByRole("button", {
    name: /Confirm preferences|确认偏好设置/,
  });
  if (await consent.isVisible().catch(() => false)) {
    const checkbox = page.getByRole("checkbox").first();
    if (await checkbox.isVisible().catch(() => false)) {
      await checkbox.uncheck().catch(() => {});
    }
    await consent.click().catch(() => {});
    await page.waitForTimeout(500);
  }

  await page
    .getByTestId("automation-setup-name")
    .fill("QA phone keep");
  await page.screenshot({ path: `${output}/01-form-filled.png` });

  // Ask Agent hands the draft to a setup conversation. If the mock stack
  // cannot create it, fall back to the standalone-form round trip.
  let conversationFlow = false;
  const askAgent = page
    .getByTestId("automation-setup-ask-agent")
    .isVisible()
    .catch(() => false);
  if (await askAgent) {
    await page.getByTestId("automation-setup-ask-agent").click();
    try {
      await page.waitForURL(/\/conversations\//, { timeout: 15000 });
      conversationFlow = true;
      await page.waitForTimeout(1500);
    } catch {
      conversationFlow = false;
    }
  }

  const viewToggle = conversationFlow
    ? page.getByTestId("automation-setup-mobile-view-conversation")
    : null;
  if (viewToggle && (await viewToggle.isVisible().catch(() => false))) {
    await viewToggle.click();
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${output}/02-conversation-view.png` });
    await page
      .getByTestId("automation-setup-mobile-view-form")
      .click();
    await page.waitForTimeout(800);
    const name = await page
      .getByTestId("automation-setup-name")
      .inputValue()
      .catch(() => "(no name field)");
    console.log(
      JSON.stringify({ mode, conversationFlow, nameAfterRoundTrip: name }, null, 2),
    );
    await page.screenshot({ path: `${output}/03-after-round-trip.png` });
  } else {
    console.log(
      JSON.stringify(
        { mode, conversationFlow, note: "conversation flow unreachable in mock" },
        null,
        2,
      ),
    );
    await page.screenshot({ path: `${output}/02-no-conversation.png` });
  }
} finally {
  await browser.close();
}
