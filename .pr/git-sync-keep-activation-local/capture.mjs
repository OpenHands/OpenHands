// Evidence capture for OHE-3927: the "Keep automation activation status
// local" switch on the Git Sync page.
//
// The mock API has no Git Sync endpoints, so this script registers the mock
// server as a cloud backend and answers two kinds of request itself:
//   - GET /api/organizations/org-1/me     an admin, who may manage Git Sync
//   - /api/automation/v1/git-sync/*       forwarded to a real automation
//                                         service that has the option
// Everything else is the unmodified mock API. The Git Sync requests go through
// Node rather than the page, so the service needs no CORS setup.
//
// Requires the mock dev server and an automation service in local mode:
//   VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3198
//   AUTOMATION_LOCAL_API_KEY=<key> ... uvicorn openhands.automation.app:app --port 18927
// Usage:
//   AUTOMATION_URL=http://127.0.0.1:18927 AUTOMATION_API_KEY=<key> \
//     node .pr/git-sync-keep-activation-local/capture.mjs [baseUrl] [outputDir]
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const baseUrl = process.argv[2] ?? "http://127.0.0.1:3198";
const output = process.argv[3] ?? ".pr/git-sync-keep-activation-local";
const serviceUrl = process.env.AUTOMATION_URL ?? "http://127.0.0.1:18927";
const apiKey = process.env.AUTOMATION_API_KEY ?? "";
await mkdir(output, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 960 } });
page.setDefaultTimeout(20000);

const saves = [];
await page.exposeFunction("automationService", async (method, path, body) => {
  const response = await fetch(`${serviceUrl}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body,
  });
  const text = await response.text();
  if (method === "PUT") saves.push({ request: body, response: text });
  return { status: response.status, body: text };
});

await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "mock-team",
        name: "Mock team organization",
        host: location.origin,
        apiKey: "mock-api-key",
        kind: "cloud",
      },
    ]),
  );
  localStorage.setItem(
    "openhands-active-backend",
    JSON.stringify({ backendId: "mock-team", orgId: "org-1" }),
  );
  localStorage.setItem("analytics-consent", "false");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem("openhands-telemetry-first-use", "true");

  const mockFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const request = new Request(input, init);
    const { pathname } = new URL(request.url);

    if (pathname === "/api/organizations/org-1/me") {
      return new Response(
        JSON.stringify({
          org_id: "org-1",
          user_id: "user-1",
          role: "admin",
          permissions: ["view_automations", "manage_automations"],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }

    if (pathname.startsWith("/api/automation/v1/git-sync/")) {
      const body = request.method === "GET" ? undefined : await request.text();
      const result = await window.automationService(
        request.method,
        pathname,
        body,
      );
      return new Response(result.body, {
        status: result.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    return mockFetch(request);
  };
});

const optionSwitch = page.getByTestId("git-sync-keep-activation-local-switch");

async function openGitSync() {
  await page.goto(`${baseUrl}/automations/git-sync`, {
    waitUntil: "domcontentloaded",
  });
  await page.getByTestId("git-sync-repo-url-input").waitFor();
  await page.waitForTimeout(1500);

  // Dismiss the analytics consent dialog when it covers the form.
  const consent = page.getByRole("button", { name: /Confirm preferences/ });
  if (await consent.isVisible().catch(() => false)) {
    await consent.click();
    await page.waitForTimeout(500);
  }
  await page.getByTestId("git-sync-save-button").scrollIntoViewIfNeeded();
}

try {
  await openGitSync();
  console.log("switch on load:", await optionSwitch.isChecked());
  await page.screenshot({ path: `${output}/01-option-off.png` });

  // The input is visually hidden; its label is what a person clicks.
  await optionSwitch.locator("xpath=..").click();
  await page.getByTestId("git-sync-save-button").click();
  await page.getByText("Git Sync settings saved.").waitFor();
  // Let the toast finish sliding in before the capture.
  await page.waitForTimeout(600);
  console.log("save request:", saves[0].request);
  console.log(
    "save response keep_activation_local:",
    JSON.parse(saves[0].response).keep_activation_local,
  );
  await page.screenshot({ path: `${output}/02-option-saved.png` });

  await openGitSync();
  console.log("switch after reload:", await optionSwitch.isChecked());
  await page.screenshot({ path: `${output}/03-option-on-after-reload.png` });
} finally {
  await browser.close();
}
