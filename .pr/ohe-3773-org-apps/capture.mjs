// Evidence capture for OHE-3773: organization-approved Apps on a cloud backend.
//
// The mock API has no organization Apps endpoints, so this script registers
// the mock server as a cloud backend and routes three kinds of request to a
// real OpenHands Enterprise backend (the harness in OpenHands/enterprise's PR
// for OHE-3773, which serves the real routes on a real database):
//   - GET /api/organizations/org-1/me                the caller's permissions
//   - /api/organizations/org-1/canvas-extensions/*   the organization's Apps
//   - GET /api/v1/web-client/config                  the mock's config, with
//                                                    `enable_canvas_apps` as
//                                                    the backend reports it
// Everything else is the unmodified mock API. The mock knows its organization
// as `org-1`; requests are rewritten to the backend's organization id. They go
// through Node rather than the page, so the backend needs no CORS setup, and
// carry the `X-Harness-User` header that picks the signed-in user.
//
// Two browser pages stay open: one for an instance Super Admin, one for a
// plain member of the same organization.
//
// Requires the mock dev server and the enterprise harness:
//   VITE_DO_NOT_TRACK=1 VITE_MOCK_API=true npx react-router dev --port 3198
//   PYTHONPATH=. uv run python .pr/ohe-3773/apps_harness.py --port 18773
// Usage:
//   node .pr/ohe-3773-org-apps/capture.mjs [baseUrl] [outputDir]
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const baseUrl = process.argv[2] ?? "http://127.0.0.1:3198";
const output = process.argv[3] ?? ".pr/ohe-3773-org-apps";
const backendUrl = process.env.OHE_URL ?? "http://127.0.0.1:18773";
const backendOrgId =
  process.env.OHE_ORG_ID ?? "0e377300-0000-4000-8000-000000000001";
const APP_SOURCE =
  "https://github.com/OpenHands/OpenHands/tree/main/src/fixtures/canvas-extensions/demo-page";
await mkdir(output, { recursive: true });

/** Calls the enterprise backend as `user`, logging what the page asked for. */
async function callBackend(user, method, path, body) {
  const response = await fetch(
    `${backendUrl}${path.replace("/organizations/org-1/", `/organizations/${backendOrgId}/`)}`,
    {
      method,
      headers: { "X-Harness-User": user, "Content-Type": "application/json" },
      body,
    },
  );
  const text = await response.text();
  if (path.includes("/canvas-extensions/") && !path.endsWith("/installed")) {
    console.log(`[${user}] ${method} ${path} -> ${response.status}`);
  }
  return {
    status: response.status,
    contentType: response.headers.get("content-type") ?? "application/json",
    body: text,
  };
}

const browser = await chromium.launch({ headless: true });

async function openCanvasAs(user) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 860 },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(30000);
  await page.exposeFunction("oheBackend", (method, path, body) =>
    callBackend(user, method, path, body),
  );
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

      if (
        pathname === "/api/organizations/org-1/me" ||
        pathname.startsWith("/api/organizations/org-1/canvas-extensions/")
      ) {
        const body =
          request.method === "GET" ? undefined : await request.text();
        const result = await window.oheBackend(request.method, pathname, body);
        return new Response(result.body, {
          status: result.status,
          headers: { "Content-Type": result.contentType },
        });
      }

      if (pathname === "/api/v1/web-client/config") {
        const [mock, real] = await Promise.all([
          mockFetch(request).then((response) => response.json()),
          window.oheBackend("GET", pathname),
        ]);
        mock.feature_flags = {
          ...mock.feature_flags,
          enable_canvas_apps:
            JSON.parse(real.body).feature_flags.enable_canvas_apps,
        };
        return new Response(JSON.stringify(mock), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      return mockFetch(request);
    };
  });
  return page;
}

async function openApps(page) {
  await page.goto(`${baseUrl}/apps`, { waitUntil: "domcontentloaded" });
  await page.getByTestId("canvas-extensions-screen").waitFor();
  await page.waitForTimeout(2500);

  // Dismiss the analytics consent dialog when it covers the page.
  const consent = page.getByRole("button", { name: /Confirm preferences/ });
  if (await consent.isVisible().catch(() => false)) {
    await consent.click();
    await page.waitForTimeout(500);
  }
}

/** What a Super Admin's switch on Instance Settings does: update, else create. */
async function setInstanceSwitch(enabled) {
  const path = "/api/admin/feature-flags/ENABLE_CANVAS_APPS";
  const updated = await callBackend(
    "super-admin",
    "PATCH",
    path,
    JSON.stringify({ enabled }),
  );
  if (updated.status === 404) {
    await callBackend(
      "super-admin",
      "POST",
      "/api/admin/feature-flags",
      JSON.stringify({
        key: "ENABLE_CANVAS_APPS",
        description: "Enable Agent Canvas Apps",
        enabled,
      }),
    );
  }
  // The backend caches a flag for a few seconds.
  await new Promise((resolve) => {
    setTimeout(resolve, 7000);
  });
}

const superAdmin = await openCanvasAs("super-admin");
const member = await openCanvasAs("member");
const card = (page) => page.getByTestId("canvas-extension-card-demo-page");
const railItem = (page) =>
  page.getByTestId("sidebar-canvas-extension-demo-page-hello");

try {
  // 1. The instance switch is off: Apps stay unavailable.
  await setInstanceSwitch(false);
  await openApps(member);
  console.log(
    "switch off, Apps in Customize:",
    await member.getByTestId("sidebar-extensions-/apps").count(),
  );
  await member.screenshot({ path: `${output}/01-switch-off.png` });

  // 2. A Super Admin turns the switch on and approves an App.
  await setInstanceSwitch(true);
  await openApps(superAdmin);
  await superAdmin.screenshot({ path: `${output}/02-super-admin-empty.png` });
  await superAdmin.getByTestId("canvas-extensions-add-button").click();
  await superAdmin
    .getByTestId("add-canvas-extension-source-input")
    .fill(APP_SOURCE);
  await superAdmin.getByTestId("add-canvas-extension-submit").click();
  await card(superAdmin).waitFor();
  await superAdmin.waitForTimeout(800);
  await superAdmin.screenshot({ path: `${output}/03-super-admin-approved.png` });

  // 3. A member sees the approved App, switches it on and opens its page.
  await openApps(member);
  await card(member).waitFor();
  console.log(
    "member controls (add, update, uninstall):",
    await member.getByTestId("canvas-extensions-add-button").count(),
    await member.getByTestId("canvas-extension-refresh-demo-page").count(),
    await member.getByTestId("canvas-extension-uninstall-demo-page").count(),
  );
  await member.screenshot({ path: `${output}/04-member-approved-list.png` });
  await card(member).getByRole("switch").click();
  await member.getByTestId("confirm-button").click();
  await railItem(member).waitFor();
  await railItem(member).click();
  await member.getByText("Hello from an extension").first().waitFor();
  await member.waitForTimeout(800);
  await member.screenshot({ path: `${output}/05-member-app-page.png` });
  console.log(
    "super admin still has it off:",
    !(await card(superAdmin).getByRole("switch").isChecked()),
  );

  // 4. The Super Admin removes the App; the member's open tab drops it.
  await superAdmin.getByTestId("canvas-extension-uninstall-demo-page").click();
  await superAdmin.getByTestId("confirm-button").click();
  await card(superAdmin).waitFor({ state: "detached" });
  const removedAt = Date.now();
  await railItem(member).waitFor({ state: "detached", timeout: 90000 });
  console.log(
    "member's open tab dropped the App after (s):",
    Math.round((Date.now() - removedAt) / 1000),
  );
  await member
    .getByText("This app is disabled, missing, or does not provide this page.")
    .waitFor();
  await member.screenshot({ path: `${output}/06-member-after-removal.png` });
} finally {
  await browser.close();
}
