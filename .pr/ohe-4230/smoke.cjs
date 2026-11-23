// Runtime check for OHE-4230 against the Canvas mock dev server.
//
// Seeds a cloud backend (a self-hosted OpenHands Enterprise instance), opens a
// conversation, and raises the error LiteLLM returns when a budget is reached,
// exactly as the conversation socket delivers it. The cloud endpoints the page
// touches are answered from inside the page, including the member's own budget
// (`/api/organizations/{org}/budgets/me`). The script saves a capture of the
// error banner and prints the text and link it shows.
//
// Usage (with `VITE_MOCK_API=true npx react-router dev --port 3057` running):
//   node .pr/ohe-4230/smoke.cjs <base url> <label> <user|org> <on|off> [output dir]
// where user/org is the budget LiteLLM reports as exceeded and on/off is
// whether the organization budget is enabled.
const path = require("node:path");
const fs = require("node:fs");
const { chromium } = require("@playwright/test");

const [
  ,
  ,
  baseUrl = "http://localhost:3057",
  label = "run",
  scope = "user",
  budgets = "on",
  outDir,
] = process.argv;
const captureDir = outDir || __dirname;
fs.mkdirSync(captureDir, { recursive: true });

const CONVERSATION_ID = "0d2fe560b4004833aec01fd5433be1c5";
const LITELLM_ERROR =
  scope === "org"
    ? "litellm.RateLimitError: RateLimitError: Litellm_proxyException - Budget has been exceeded! Team=test-org Current cost: 2913.4128, Max budget: 2913.1026"
    : "litellm.RateLimitError: RateLimitError: Litellm_proxyException - Budget has been exceeded! TeamMember=user-1:test-org Current cost: 348.96549289999996, Max budget: 348.9626492500001";

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "dark",
  });
  await context.addInitScript(
    ({ origin, conversationId, budgetsEnabled }) => {
      localStorage.setItem("openhands-onboarded", "1");
      localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "ohe",
            name: "OHE",
            host: origin,
            apiKey: "cloud-test-api-key",
            kind: "cloud",
          },
        ]),
      );
      localStorage.setItem(
        "openhands-active-backend",
        JSON.stringify({ backendId: "ohe", orgId: "test-org" }),
      );
      localStorage.setItem("analytics-consent", "false");
      localStorage.setItem("openhands-telemetry-consent", "denied");
      localStorage.setItem("openhands-telemetry-first-use", "true");

      const reply = (status, body) =>
        new Response(JSON.stringify(body), {
          status,
          headers: { "content-type": "application/json" },
        });
      const conversation = {
        id: conversationId,
        created_by_user_id: "user-1",
        sandbox_id: "sandbox-1",
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        title: "Exhaustive Repository Analysis",
        trigger: null,
        pr_number: [],
        llm_model: "openhands/claude-opus",
        metrics: null,
        created_at: "2026-11-16T12:00:00Z",
        updated_at: "2026-11-16T12:30:00Z",
        sandbox_status: "PAUSED",
        execution_status: "error",
        conversation_url: null,
        session_api_key: null,
        sub_conversation_ids: [],
        public: false,
      };
      const originalFetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        let url = input;
        if (input instanceof URL) url = input.href;
        else if (typeof input !== "string") url = input.url;
        const { pathname } = new URL(url, window.location.origin);

        if (pathname === "/api/v1/web-client/config") {
          return reply(200, {
            feature_flags: { deployment_mode: "self_hosted" },
          });
        }
        if (pathname === "/api/v1/app-conversations/search") {
          return reply(200, { items: [conversation], next_page_id: null });
        }
        if (pathname === "/api/v1/app-conversations") {
          return reply(200, [conversation]);
        }
        if (pathname === "/api/v1/skills/search") {
          return reply(200, { items: [], next_page_id: null });
        }
        if (pathname === "/api/organizations/test-org/me") {
          return reply(200, {
            org_id: "test-org",
            user_id: "user-1",
            role: "member",
            permissions: ["view_org_settings"],
          });
        }
        if (pathname === "/api/organizations/test-org/budgets/me") {
          return reply(
            200,
            budgetsEnabled
              ? {
                  enabled: true,
                  monthly_limit: 500,
                  org_monthly_limit: 2000,
                  limit_reached_instructions:
                    "Contact #it-help to request a higher budget.\nBudgets reset on the 1st of each month.",
                }
              : { enabled: false },
          );
        }
        return originalFetch(input, init);
      };
    },
    {
      origin: baseUrl,
      conversationId: CONVERSATION_ID,
      budgetsEnabled: budgets === "on",
    },
  );

  const page = await context.newPage();
  await page.goto(`${baseUrl}/conversations/${CONVERSATION_ID}`, {
    waitUntil: "domcontentloaded",
  });
  // Vite optimises deps on the first request and reloads once.
  await page.waitForTimeout(8000);
  await page.reload({ waitUntil: "domcontentloaded" });
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.isVisible({ timeout: 5000 }).catch(() => false)) {
    await consent.click();
  }
  await page.waitForTimeout(6000);

  // What the conversation socket does with a ConversationErrorEvent.
  await page.evaluate(async (detail) => {
    const store = await import("/src/stores/error-message-store.ts");
    store.useErrorMessageStore
      .getState()
      .setErrorMessage(detail, "conversation", "LLMRateLimitError", {
        kind: "rate_limit",
        retryable: true,
        user_action: "retry",
      });
  }, LITELLM_ERROR);

  const banner = page.getByTestId("error-message-banner");
  const shown = await banner
    .waitFor({ timeout: 10000 })
    .then(() => true)
    .catch(() => false);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(captureDir, `${label}.png`) });

  const link = page.getByTestId("error-message-banner-link");
  const hasLink = shown && (await link.count()) > 0;
  console.log(
    JSON.stringify(
      {
        label,
        exceeded: scope,
        organizationBudget: budgets,
        url: page.url(),
        bannerShown: shown,
        bannerText: shown
          ? await page.getByTestId("error-message-banner-content").innerText()
          : null,
        link: hasLink
          ? {
              text: await link.innerText(),
              href: await link.getAttribute("href"),
            }
          : null,
      },
      null,
      2,
    ),
  );
  await browser.close();
})();
