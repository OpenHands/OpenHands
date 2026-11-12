// Runtime check for OHE-4150 against the Canvas mock dev server.
//
// Seeds a cloud backend whose owner still has an activated personal agent
// profile ("reviewer", LLM profile "Smart") while the org's active LLM profile
// is "Fast". The cloud endpoints the home model picker touches are answered
// from inside the page. The script picks the LLM profile the pill is not
// showing, then reports which write the pick made, whether the "couldn't be
// updated" toast appeared, and saves a capture of the home page.
//
// Usage (with `VITE_MOCK_API=true npx react-router dev --port 3027` running):
//   node .pr/ohe-4150/smoke.cjs <base url> <label> <on|off> [output dir]
// where on/off is the instance's `enable_agent_profiles_page` flag.
const path = require("node:path");
const fs = require("node:fs");
const { chromium } = require("@playwright/test");

const [, , baseUrl = "http://localhost:3027", label = "run", flagArg, outDir] =
  process.argv;
const agentProfilesEnabled = flagArg === "on";
const captureDir = outDir || __dirname;
fs.mkdirSync(captureDir, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "dark",
  });
  // Lives on the Node side, so it survives Vite's first-load page reload.
  const writes = [];
  await context.exposeFunction("__recordWrite", (entry) => writes.push(entry));
  await context.addInitScript(
    ({ origin, enabled }) => {
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
      const activeLlmProfile = () =>
        sessionStorage.getItem("active-llm-profile") || "Fast";
      const originalFetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        let url = input;
        if (input instanceof URL) url = input.href;
        else if (typeof input !== "string") url = input.url;
        const method = (
          (init && init.method) ||
          (typeof input === "object" && input.method) ||
          "GET"
        ).toUpperCase();
        const { pathname } = new URL(url, window.location.origin);
        const record = () => window.__recordWrite(`${method} ${pathname}`);

        if (pathname === "/api/v1/web-client/config") {
          return reply(200, {
            feature_flags: {
              deployment_mode: "self_hosted",
              enable_agent_profiles_page: enabled,
            },
          });
        }
        // The mock API does not serve these cloud lists; left unanswered,
        // their 502s raise an unrelated "An error occurred" toast.
        if (
          pathname === "/api/v1/app-conversations/search" ||
          pathname === "/api/v1/skills/search"
        ) {
          return reply(200, { items: [], next_page_id: null });
        }
        if (pathname === "/api/organizations/test-org/me") {
          return reply(200, {
            org_id: "test-org",
            user_id: "user-1",
            role: "owner",
            permissions: ["view_org_settings", "edit_org_settings"],
          });
        }
        if (pathname === "/api/organizations/test-org/profiles") {
          return reply(200, {
            profiles: [
              {
                name: "Fast",
                model: "openai/gpt-4o-mini",
                base_url: null,
                api_key_set: true,
              },
              {
                name: "Smart",
                model: "anthropic/claude-opus",
                base_url: null,
                api_key_set: true,
              },
            ],
            active_profile: activeLlmProfile(),
          });
        }
        const activate = pathname.match(
          /^\/api\/organizations\/test-org\/profiles\/([^/]+)\/activate$/,
        );
        if (activate && method === "POST") {
          record();
          sessionStorage.setItem("active-llm-profile", activate[1]);
          return reply(200, {
            name: activate[1],
            message: `Profile '${activate[1]}' activated`,
            llm: { model: "stub" },
          });
        }
        if (pathname === "/api/agent-profiles" && method === "GET") {
          return reply(200, {
            profiles: [
              {
                id: "p1",
                name: "reviewer",
                agent_kind: "openhands",
                revision: 1,
                llm_profile_ref: "Smart",
                scope: "personal",
              },
            ],
            active_agent_profile_id: "p1",
          });
        }
        if (pathname === "/api/agent-profiles/reviewer") {
          record();
          if (method === "GET") {
            return reply(200, {
              name: "reviewer",
              scope: "personal",
              profile: {
                id: "p1",
                name: "reviewer",
                agent_kind: "openhands",
                revision: 1,
                llm_profile_ref: "Smart",
              },
            });
          }
          // The app-server's answer to a save without `scope=personal` for a
          // name that lives in a member's personal collection.
          return reply(409, {
            detail:
              "Agent profile 'reviewer' already exists in a member's personal profiles",
          });
        }
        return originalFetch(input, init);
      };
    },
    { origin: baseUrl, enabled: agentProfilesEnabled },
  );

  const page = await context.newPage();
  await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded" });
  // Vite optimises deps on the first request and reloads once.
  await page.waitForTimeout(8000);
  await page.reload({ waitUntil: "domcontentloaded" });
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.isVisible({ timeout: 5000 }).catch(() => false)) {
    await consent.click();
  }
  const pill = page.getByTestId("chat-input-llm-profile");
  await pill.waitFor({ timeout: 30000 });
  // Let the profile lists, the membership and the instance config settle.
  await page.waitForTimeout(4000);

  const pillBeforePick = (await pill.innerText()).trim();
  const picked = pillBeforePick.includes("Smart") ? "Fast" : "Smart";
  writes.length = 0;
  await pill.click();
  await page.getByTestId(`chat-input-llm-profile-option-${picked}`).click();
  const toastShown = await page
    .getByText("couldn't be updated to match")
    .first()
    .waitFor({ timeout: 4000 })
    .then(() => true)
    .catch(() => false);
  await page.waitForTimeout(500);
  const pillAfterPick = (await pill.innerText()).trim();
  await page.screenshot({ path: path.join(captureDir, `${label}.png`) });

  console.log(
    JSON.stringify(
      {
        label,
        enable_agent_profiles_page: agentProfilesEnabled,
        pillBeforePick,
        picked,
        requestsAfterPick: writes,
        toastShown,
        pillAfterPick,
      },
      null,
      2,
    ),
  );
  await browser.close();
})();
