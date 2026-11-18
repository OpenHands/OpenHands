// Evidence capture for #19061: the replacement card for an existing secret.
// Requires the MSW dev server: VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3193
// Usage: node .pr/bugs/secret-replacement/capture.mjs after <outputDir>
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const mode = process.argv[2] ?? "after";
const output = process.argv[3] ?? `.pr/bugs/secret-replacement/${mode}`;
await mkdir(output, { recursive: true });

const CONVERSATION_ID = "2";

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
  // An existing secret the user saved earlier (mock backend state).
  await page.goto(`http://127.0.0.1:3193/conversations/${CONVERSATION_ID}`, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForTimeout(2000);
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

  await page.evaluate(async () => {
    await fetch("/api/settings/secrets", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: "EXAMPLE_SERVICE_API_KEY",
        value: "stored-old-value",
        description: "Saved earlier for the factory service",
      }),
    });
  });

  // The agent re-requests the same name: the tool call is claimed and the
  // pending request is queued as a replacement (the store is persisted, so
  // seeding it reproduces exactly what the interception queues).
  await page.evaluate((conversationId) => {
    localStorage.setItem(
      "openhands-pending-secret-requests",
      JSON.stringify({
        state: {
          requests: [
            {
              conversationId,
              toolCallId: "tool-call-replacement",
              name: "EXAMPLE_SERVICE_API_KEY",
              reason:
                "Rotate the factory service credential; the old one was revoked.",
              existing: true,
            },
          ],
        },
        version: 0,
      }),
    );
  }, CONVERSATION_ID);
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.getByTestId("secret-request-card").waitFor();
  await page.waitForTimeout(500);
  const consent2 = page.getByRole("button", {
    name: /Confirm preferences|确认偏好设置/,
  });
  if (await consent2.isVisible().catch(() => false)) {
    const checkbox2 = page.getByRole("checkbox").first();
    if (await checkbox2.isVisible().catch(() => false)) {
      await checkbox2.uncheck().catch(() => {});
    }
    await consent2.click().catch(() => {});
    await page.waitForTimeout(500);
  }
  await page.waitForTimeout(500);

  const cardText = await page
    .getByTestId("secret-request-card")
    .innerText();
  const valueState = await page.evaluate(() => {
    const input = document.querySelector('[data-testid="value-input"]');
    return input
      ? { type: input.getAttribute("type"), value: input.value }
      : null;
  });
  console.log(
    JSON.stringify({ mode, cardText, valueState }, null, 2),
  );
  await page
    .getByTestId("secret-request-card")
    .screenshot({ path: `${output}/01-replacement-card.png` });

  // The save stays disabled until a new value is typed, so a "saved" report
  // can never happen without a real replacement.
  const blankDisabled = await page
    .getByTestId("edit-secret-form")
    .getByTestId("submit-button")
    .isDisabled();
  console.log(JSON.stringify({ blankSaveDisabled: blankDisabled }, null, 2));
  await page
    .getByTestId("secret-request-card")
    .screenshot({ path: `${output}/02-save-disabled-blank.png` });

  // Typing the new value enables the save; it goes through the update path
  // and completes the request with the secret's identity.
  const form = page.getByTestId("edit-secret-form");
  await page.fill('[data-testid="value-input"]', "rotated-new-value");
  await form.getByTestId("submit-button").click();
  await page.waitForTimeout(1200);
  const afterSave = await page.evaluate(() => ({
    cardClosed: !document.querySelector('[data-testid="secret-request-card"]'),
    followUp: document.body.textContent.includes("EXAMPLE_SERVICE_API_KEY"),
  }));
  console.log(JSON.stringify({ afterSave }, null, 2));
  await page.screenshot({ path: `${output}/03-saved-and-reported.png` });
} finally {
  await browser.close();
}
