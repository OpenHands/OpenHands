/**
 * Mock-LLM E2E: MCP "Test connection" credential verification (PR #1175).
 *
 * Background: the Slack MCP server lists its tools with *any* credentials and
 * only exercises them inside tool handlers, so the old connection test
 * (`tools/list` only) reported "Connected — 8 tool(s) available" for an
 * invalid Team ID / Bot token. This suite proves the fix end-to-end through
 * the real browser → real agent-server stack, with only `POST /api/mcp/test`
 * intercepted so we can drive the agent server's `tool_result` payload (the
 * credential interpretation itself runs client-side in `McpService`).
 *
 * Verifies the PR's acceptance criteria:
 *   - Installing Slack with invalid credentials shows
 *     "Credential check failed: invalid_auth" and does NOT install.
 *   - A read-only `slack_list_channels {limit: 1}` probe is attached to the
 *     test request for catalog servers (Slack) — and to NOTHING for custom
 *     (non-catalog) servers.
 *   - A valid token merely missing a scope (`missing_scope`) is NOT treated
 *     as a credential failure — the install still succeeds.
 *   - Valid credentials connect successfully.
 *   - Older agent servers that omit `tool_result` behave exactly as before.
 *   - The Edit modal's "Test connection" verifies the *stored* credentials —
 *     it never tests the literal redaction placeholder the browser sees.
 */

import {
  test,
  expect,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  ensureMockLLMProfileViaAPI,
} from "../utils/mock-llm-helpers";

// The read-only verification probe the service attaches for Slack.
const SLACK_TOOL_CALL = {
  name: "slack_list_channels",
  arguments: { limit: 1 },
};

// Placeholder the settings API substitutes for secret env values when read
// without X-Expose-Secrets (the MCP page's mode). The whole point of the fix
// is that the test request must NOT carry this value.
const REDACTED = "**********";

/** The JSON body the GUI POSTs to `/api/mcp/test` (captured via interception). */
interface McpTestRequestBody {
  server?: { type?: string; url?: string; env?: Record<string, string> };
  name?: string;
  timeout?: number;
  tool_call?: { name: string; arguments: Record<string, unknown> };
}

const jsonRoute = (body: unknown) => ({
  status: 200,
  contentType: "application/json",
  body: JSON.stringify(body),
});

/**
 * Shape the patched agent server returns from `POST /api/mcp/test`: tools were
 * listed successfully and the attached `tool_call` ran, with its raw outcome
 * in `tool_result`. `McpService` interprets this client-side.
 */
const slackToolResult = (text: string, isError = false) => ({
  ok: true,
  tools: ["slack_list_channels", "slack_post_message"],
  tool_result: { is_error: isError, text },
});

async function getSettings(request: APIRequestContext) {
  const resp = await request.get(`${BACKEND_URL}/api/settings`, {
    headers: { "X-Session-API-Key": SESSION_API_KEY },
  });
  expect(resp.ok(), `GET /api/settings: ${resp.status()}`).toBe(true);
  return resp.json();
}

/** Detect the Slack server in persisted settings. */
function hasSlackServer(settings: unknown): boolean {
  const servers = (settings as { agent_settings?: { mcp_config?: unknown } })
    ?.agent_settings?.mcp_config;
  if (!servers) return false;
  return JSON.stringify(servers).includes("@zencoderai/slack-mcp-server");
}

async function patchMcpConfig(request: APIRequestContext, mcpConfig: unknown) {
  const resp = await request.patch(`${BACKEND_URL}/api/settings`, {
    headers: {
      "X-Session-API-Key": SESSION_API_KEY,
      "Content-Type": "application/json",
    },
    data: { agent_settings_diff: { mcp_config: mcpConfig } },
  });
  expect(resp.ok(), `seed mcp_config: ${resp.status()}`).toBe(true);
}

/** Seed a Slack stdio server (matches the marketplace catalog entry). */
async function installSlackViaAPI(
  request: APIRequestContext,
  env: Record<string, string>,
) {
  await patchMcpConfig(request, {
    slack: {
      command: "npx",
      args: ["-y", "@zencoderai/slack-mcp-server"],
      env,
    },
  });
}

async function openSlackInstallModal(page: Page) {
  await page.goto("/mcp", { waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  await expect(page.getByTestId("mcp-marketplace-grid")).toBeVisible({
    timeout: 10_000,
  });
  await page.getByTestId("mcp-marketplace-card-slack").click();
  const modal = page.getByTestId("mcp-install-modal");
  await expect(modal).toBeVisible({ timeout: 5_000 });
  await expect(modal).toHaveAttribute("data-marketplace-id", "slack");
  return modal;
}

async function fillSlackCredentials(
  page: Page,
  teamId: string,
  botToken: string,
) {
  await page.getByTestId("mcp-install-field-SLACK_TEAM_ID").fill(teamId);
  await page.getByTestId("mcp-install-field-SLACK_BOT_TOKEN").fill(botToken);
}

/** Open the editor for the first installed server (clicking its title avoids
 *  the delete toggle in the card's top-right corner). */
async function openInstalledEditor(page: Page) {
  await page.goto("/mcp", { waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  const list = page.getByTestId("mcp-installed-list");
  await expect(list).toBeVisible({ timeout: 10_000 });
  const item = list.getByTestId("mcp-server-item").first();
  await expect(item).toBeVisible();
  await item.locator("h3").click();
  await expect(page.getByTestId("mcp-custom-editor")).toBeVisible({
    timeout: 5_000,
  });
}

test.describe.configure({ mode: "serial" });

test.describe("MCP Test Connection credential verification (Slack)", () => {
  test.beforeEach(async ({ page, request }) => {
    await seedLocalStorage(page);
    // A configured LLM profile keeps the MCP page free of "not configured"
    // banners that could intercept clicks. Done via API for speed.
    await ensureMockLLMProfileViaAPI(request);
  });

  test.afterEach(async ({ request }) => {
    // Reset MCP config so each test starts from a clean installed list.
    await patchMcpConfig(request, null).catch(() => {});
  });

  test("install: invalid Slack credentials are blocked with a credential-check error", async ({
    page,
    request,
  }) => {
    await routeSessionApiKey(page);

    const testRequests: McpTestRequestBody[] = [];
    await page.route("**/api/mcp/test", async (route) => {
      testRequests.push(route.request().postDataJSON());
      await route.fulfill(
        jsonRoute(slackToolResult('{"ok":false,"error":"invalid_auth"}')),
      );
    });

    const modal = await openSlackInstallModal(page);
    await fillSlackCredentials(page, "T_INVALID", "xoxb-invalid-token");
    await page.getByTestId("mcp-install-submit").click();

    // The credential failure is surfaced and the modal stays open.
    const error = page.getByTestId("mcp-install-modal-error");
    await expect(error).toBeVisible({ timeout: 10_000 });
    await expect(error).toContainText("Credential check failed: invalid_auth");
    await expect(modal).toBeVisible();

    // The read-only verification probe was attached to the test request.
    expect(testRequests[0]?.tool_call).toEqual(SLACK_TOOL_CALL);

    // Nothing was installed.
    expect(hasSlackServer(await getSettings(request))).toBe(false);
  });

  test("install: a valid token missing only a scope still installs (missing_scope is not a credential failure)", async ({
    page,
    request,
  }) => {
    await routeSessionApiKey(page);
    await page.route("**/api/mcp/test", async (route) => {
      await route.fulfill(
        jsonRoute(slackToolResult('{"ok":false,"error":"missing_scope"}')),
      );
    });

    const modal = await openSlackInstallModal(page);
    await fillSlackCredentials(
      page,
      "T0123456",
      "xoxb-valid-but-missing-scope",
    );
    await page.getByTestId("mcp-install-submit").click();

    // missing_scope proves the token authenticated → install proceeds.
    await expect(modal).not.toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId("mcp-installed-list")).toBeVisible({
      timeout: 10_000,
    });
    expect(hasSlackServer(await getSettings(request))).toBe(true);
  });

  test("install: an older agent server that omits tool_result still installs (compat)", async ({
    page,
    request,
  }) => {
    await routeSessionApiKey(page);
    // No `tool_result` field → the response passes through uninterpreted.
    await page.route("**/api/mcp/test", async (route) => {
      await route.fulfill(jsonRoute({ ok: true, tools: ["a", "b"] }));
    });

    const modal = await openSlackInstallModal(page);
    await fillSlackCredentials(page, "T0123456", "xoxb-some-token");
    await page.getByTestId("mcp-install-submit").click();

    await expect(modal).not.toBeVisible({ timeout: 10_000 });
    expect(hasSlackServer(await getSettings(request))).toBe(true);
  });

  test("edit: Test Connection verifies the stored credentials and surfaces a credential failure", async ({
    page,
    request,
  }) => {
    await installSlackViaAPI(request, {
      SLACK_TEAM_ID: "T0123456",
      SLACK_BOT_TOKEN: "xoxb-stored-secret",
    });
    await routeSessionApiKey(page);

    const testRequests: McpTestRequestBody[] = [];
    await page.route("**/api/mcp/test", async (route) => {
      testRequests.push(route.request().postDataJSON());
      await route.fulfill(
        jsonRoute(slackToolResult('{"ok":false,"error":"invalid_auth"}')),
      );
    });

    await openInstalledEditor(page);
    await page.getByTestId("mcp-test-connection").click();

    const message = page.getByTestId("mcp-test-message");
    await expect(message).toBeVisible({ timeout: 10_000 });
    await expect(message).toContainText(
      "Credential check failed: invalid_auth",
    );

    // The Slack verification probe is attached in the edit flow too...
    expect(testRequests[0]?.tool_call).toEqual(SLACK_TOOL_CALL);
    // ...and the request tests the *stored* token, never the placeholder the
    // browser sees in the redacted form (the second root cause the PR fixes).
    expect(testRequests[0]?.server?.env?.SLACK_BOT_TOKEN).not.toBe(REDACTED);
  });

  test("edit: Test Connection reports success for valid stored credentials", async ({
    page,
    request,
  }) => {
    await installSlackViaAPI(request, {
      SLACK_TEAM_ID: "T0123456",
      SLACK_BOT_TOKEN: "xoxb-stored-secret",
    });
    await routeSessionApiKey(page);
    await page.route("**/api/mcp/test", async (route) => {
      await route.fulfill(
        jsonRoute(slackToolResult('{"ok":true,"channels":[]}')),
      );
    });

    await openInstalledEditor(page);
    await page.getByTestId("mcp-test-connection").click();

    const message = page.getByTestId("mcp-test-message");
    await expect(message).toBeVisible({ timeout: 10_000 });
    // 2 tools listed → "Connected — 2 tool(s) available".
    await expect(message).toContainText("2 tool(s) available");
    await expect(message).not.toContainText("Credential check failed");
  });

  test("custom (non-catalog) server: Test Connection attaches no verification probe", async ({
    page,
    request,
  }) => {
    // A custom server is not in the marketplace catalog → no credential probe.
    await patchMcpConfig(request, {
      my_custom: { url: "https://custom.example.test/mcp" },
    });
    await routeSessionApiKey(page);

    const testRequests: McpTestRequestBody[] = [];
    await page.route("**/api/mcp/test", async (route) => {
      testRequests.push(route.request().postDataJSON());
      await route.fulfill(jsonRoute({ ok: true, tools: ["search"] }));
    });

    await openInstalledEditor(page);
    await page.getByTestId("mcp-test-connection").click();

    const message = page.getByTestId("mcp-test-message");
    await expect(message).toBeVisible({ timeout: 10_000 });
    await expect(message).toContainText("1 tool(s) available");

    // Behavior unchanged for custom servers: no tool_call attached.
    expect(testRequests[0]?.tool_call).toBeUndefined();
  });
});
