// Evidence capture for OHE-4206: the Visibility setting on an automation that
// runs as a person.
//
// The setting is offered only in a team organization on a cloud backend whose
// automation service advertises `automationVisibility`. The mock API has
// neither, so this script registers the mock server as a cloud backend and
// answers three requests the way such a deployment would:
//   - GET /api/organizations/org-1/me        a member of a team organization
//   - GET /api/automation/v1/capabilities    the mock's, plus the new feature
//   - an automation the form created         returned with its `visibility`
// Everything else is the unmodified mock API.
//
// Requires the mock dev server:
//   VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3197
// Usage: node .pr/automation-visibility/capture.mjs [baseUrl] [outputDir]
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const baseUrl = process.argv[2] ?? "http://127.0.0.1:3197";
const output = process.argv[3] ?? ".pr/automation-visibility";
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

  const json = (body, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  const visibilityById = new Map();
  const mockFetch = window.fetch.bind(window);

  window.fetch = async (input, init) => {
    const request = new Request(input, init);
    const { pathname } = new URL(request.url);

    if (pathname === "/api/organizations/org-1/me") {
      return json({
        org_id: "org-1",
        user_id: "user-1",
        role: "member",
        permissions: ["view_automations"],
      });
    }

    if (pathname === "/api/automation/v1/capabilities") {
      const capabilities = await (await mockFetch(request)).json();
      return json({
        ...capabilities,
        features: [...(capabilities.features ?? []), "automationVisibility"],
      });
    }

    const created =
      request.method === "POST" &&
      /^\/api\/automation\/v1(\/preset\/(prompt|plugin))?$/.test(pathname);
    const requested = created ? await request.clone().json() : null;
    const response = await mockFetch(request);
    const automationId = pathname.match(
      /^\/api\/automation\/v1\/([0-9a-f-]{36})$/,
    )?.[1];
    if (!response.ok || !(created || automationId)) return response;

    const automation = await response.clone().json();
    if (created) {
      window.createRequest = `POST ${pathname} ${JSON.stringify(requested)}`;
      visibilityById.set(automation.id, requested.visibility ?? "organization");
    }
    const visibility = visibilityById.get(automation.id);
    if (!visibility) return response;
    return json(
      { ...automation, visibility, run_as_kind: "user" },
      response.status,
    );
  };
});

try {
  await page.goto(`${baseUrl}/automations/setup`, {
    waitUntil: "domcontentloaded",
  });
  await page.getByTestId("automation-setup-name").waitFor();
  await page.waitForTimeout(2000);

  // Dismiss the analytics consent dialog when it covers the form.
  const consent = page.getByRole("button", { name: /Confirm preferences/ });
  if (await consent.isVisible().catch(() => false)) {
    await consent.click();
    await page.waitForTimeout(500);
  }

  await page
    .getByTestId("automation-setup-name")
    .fill("Daily conversion report");
  await page
    .getByTestId("automation-setup-prompt")
    .fill("Summarize yesterday's conversion numbers.");

  const visibility = page.getByTestId("automation-setup-visibility");
  await visibility.scrollIntoViewIfNeeded();
  console.log(
    "default visibility:",
    await visibility.inputValue(),
    "| options:",
    await visibility.locator("option").allTextContents(),
  );
  await page.screenshot({
    path: `${output}/01-setup-default-organization.png`,
  });

  await visibility.selectOption("private");
  await page.screenshot({ path: `${output}/02-setup-private-selected.png` });

  await page.getByTestId("automation-setup-create").click({ force: true });
  await page.waitForURL(/\/automations\/[0-9a-f-]{36}/, { timeout: 15000 });
  await page.getByTestId("automation-settings-section").waitFor();
  await page.waitForTimeout(1000);
  console.log(
    "create request:",
    await page.evaluate(() => window.createRequest),
  );
  await page
    .getByTestId("automation-settings-section")
    .scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/03-detail-private.png` });
  console.log(
    "detail settings:",
    (await page.getByTestId("automation-settings-section").innerText())
      .split("\n")
      .filter(Boolean)
      .join(" | "),
  );
} finally {
  await browser.close();
}
