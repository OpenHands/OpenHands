import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, copyFile, writeFile, readFile, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const artifacts = resolve(root, ".pr");
const canvasRevision = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: root,
  encoding: "utf8",
}).trim();
const buildIndex = resolve(root, "build/index.html");
const buildHash = createHash("sha256")
  .update(await readFile(buildIndex))
  .digest("hex");
const buildModifiedAt = (await stat(buildIndex)).mtime.toISOString();
const scratch = resolve(root, ".agent_tmp/onboarding-evidence/recordings");
const origin = process.env.ONBOARDING_ORIGIN ?? "http://127.0.0.1:18736";
const sessionKey = process.env.ONBOARDING_SESSION_API_KEY ?? "fixture-session";
const fixture = process.env.ONBOARDING_EXISTING_ACCOUNT !== "1";
if (!fixture)
  assert.ok(
    process.env.ONBOARDING_SESSION_API_KEY,
    "Existing-account recording requires an explicitly supplied server session key",
  );
const prefix = fixture
  ? "onboarding-auth-fixture"
  : "onboarding-existing-login";
await mkdir(scratch, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 1100 },
  recordVideo: { dir: scratch, size: { width: 1440, height: 1100 } },
});
await context.grantPermissions(["clipboard-read", "clipboard-write"], {
  origin,
});
const page = await context.newPage();
const errors = [];
const requests = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("response", (response) => {
  const url = new URL(response.url());
  if (
    url.origin === origin &&
    url.pathname.startsWith("/api/") &&
    (url.pathname.startsWith("/api/acp/codex/auth") ||
      url.pathname.includes("agent-profiles"))
  ) {
    requests.push({
      method: response.request().method(),
      path: url.pathname,
      status: response.status(),
    });
  }
});
context.on("page", (popup) => {
  if (popup !== page) void popup.close();
});
await page.route(/(?:posthog\.com|z\.openhands\.dev)/, (route) =>
  route.abort(),
);
await page.addInitScript(
  ({ sessionKey, fixture }) => {
    localStorage.removeItem("openhands-onboarded");
    localStorage.setItem("i18nextLng", "en");
    localStorage.setItem("analytics-consent", "false");
    localStorage.setItem("openhands-telemetry-consent", "denied");
    localStorage.setItem("openhands-telemetry-first-use", "true");
    localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: "onboarding-evidence",
          name: fixture
            ? "Mock OpenAI transport fixture"
            : "Real server: existing Codex login",
          host: location.origin,
          apiKey: sessionKey,
          kind: "local",
        },
      ]),
    );
    localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: "onboarding-evidence", orgId: null }),
    );
    document.addEventListener("DOMContentLoaded", () => {
      const addLabel = () => {
        if (document.getElementById("onboarding-evidence-label")) return;
        const label = document.createElement("div");
        label.id = "onboarding-evidence-label";
        label.textContent = fixture
          ? "ONBOARDING DEMO · Real Canvas + Agent Server · Mock OpenAI authentication transport · No model request"
          : "ONBOARDING DEMO · Real Canvas + Agent Server · Previously authorized Codex account · No new consent/model request";
        label.style.cssText =
          "position:fixed;top:0;left:0;right:0;z-index:2147483647;padding:9px 12px;background:#fff4ce;color:#242424;font:600 15px sans-serif;text-align:center;pointer-events:none";
        document.body.appendChild(label);
      };
      addLabel();
      new MutationObserver(addLabel).observe(document.documentElement, {
        childList: true,
        subtree: true,
      });
    });
  },
  { sessionKey, fixture },
);
async function waitForActiveOnboardingStep(testId) {
  await page.waitForFunction((id) => {
    const target = document.querySelector('[data-testid="' + id + '"]');
    const rail = document.querySelector(
      '[data-testid="onboarding-slide-rail"]',
    );
    const slide = target?.closest('[data-testid^="onboarding-slide-"]');
    return Boolean(
      rail &&
      slide &&
      slide.getAttribute("data-testid") ===
        "onboarding-slide-" + rail.getAttribute("data-current-step") &&
      Math.abs(new DOMMatrixReadOnly(getComputedStyle(slide).transform).m41) <
        0.5,
    );
  }, testId);
}
async function settleOnboardingSlide() {
  await page.waitForFunction(() => {
    const rail = document.querySelector(
      '[data-testid="onboarding-slide-rail"]',
    );
    if (!rail) return false;
    const slide = document.querySelector(
      '[data-testid="onboarding-slide-' +
        rail.getAttribute("data-current-step") +
        '"]',
    );
    if (!slide) return false;
    return (
      Math.abs(new DOMMatrixReadOnly(getComputedStyle(slide).transform).m41) <
      0.5
    );
  });
}
let video;
try {
  const setup = await page.request.patch(origin + "/api/settings", {
    headers: { "X-Session-API-Key": sessionKey },
    data: {
      misc_settings_diff: {
        app_preferences: { language: "en", user_consents_to_analytics: false },
      },
    },
  });
  assert.equal(setup.ok(), true, "Backend preference setup failed");
  if (fixture) {
    await page.request.post(origin + "/api/acp/codex/auth/logout", {
      headers: { "X-Session-API-Key": sessionKey },
    });
  }
  await page.goto(origin + "/conversations", { waitUntil: "domcontentloaded" });
  await waitForActiveOnboardingStep("onboarding-step-choose-agent");
  await page.getByTestId("onboarding-agent-option-codex").click();
  await settleOnboardingSlide();
  await page.screenshot({
    path: resolve(artifacts, prefix + "-choose-agent.png"),
  });
  await page.waitForTimeout(1200);
  await page.getByTestId("onboarding-agent-next").click();
  const card = page
    .getByTestId("onboarding-step-setup-acp-secrets")
    .getByTestId("codex-auth-card");
  await waitForActiveOnboardingStep("onboarding-step-setup-acp-secrets");
  await card.waitFor({ state: "visible" });
  assert.equal(
    await page.getByTestId("onboarding-acp-secret-OPENAI_API_KEY").inputValue(),
    "",
  );
  if (fixture) {
    await card
      .getByRole("button", { name: "Sign in with ChatGPT", exact: true })
      .click();
    await card.getByText(/^DEMO-ONLY-\d+$/).waitFor();
    await settleOnboardingSlide();
    await page.screenshot({
      path: resolve(artifacts, prefix + "-pending.png"),
    });
    await page.waitForTimeout(1200);
    await card.getByTestId("copy-to-clipboard").click();
    await card.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.waitForTimeout(700);
    await card
      .getByRole("button", { name: "Sign in with ChatGPT", exact: true })
      .click();
  }
  await card
    .getByRole("button", { name: "Disconnect", exact: true })
    .waitFor({ timeout: 30000 });
  await settleOnboardingSlide();
  await page.screenshot({
    path: resolve(artifacts, prefix + "-connected.png"),
  });
  await page.waitForTimeout(1500);
  const next = page.getByTestId("onboarding-acp-secrets-next");
  await next.waitFor();
  assert.equal(
    await next.isEnabled(),
    true,
    "Connected Codex must allow onboarding to advance",
  );
  await next.click();
  await waitForActiveOnboardingStep("onboarding-step-say-hello");
  await settleOnboardingSlide();
  await page.screenshot({ path: resolve(artifacts, prefix + "-ready.png") });
  await page.waitForTimeout(1500);
  await page.getByTestId("onboarding-hello-close").click();
  await page.getByTestId("onboarding-modal").waitFor({ state: "hidden" });
  const statusResponse = await page.request.get(
    origin + "/api/acp/codex/auth/status",
    { headers: { "X-Session-API-Key": sessionKey } },
  );
  assert.equal((await statusResponse.json()).connected, true);
  const settingsResponse = await page.request.get(origin + "/api/settings", {
    headers: { "X-Session-API-Key": sessionKey },
  });
  const settings = await settingsResponse.json();
  assert.equal(settings.agent_settings.agent_kind, "acp");
  assert.equal(settings.agent_settings.acp_server, "codex");
  assert.deepEqual(errors, []);
  await writeFile(
    resolve(artifacts, prefix + "-results.json"),
    JSON.stringify(
      {
        canvas_revision: canvasRevision,
        build_index_sha256: buildHash,
        build_index_modified_at: buildModifiedAt,
        recorded_at: new Date().toISOString(),
        backend: "Packaged Agent Server from SDK PR #5452",
        authentication: fixture
          ? "Mock OpenAI transport; nonfunctional test tokens"
          : "Existing human-authorized Codex account; no login initiation recorded",
        first_run_browser_context: true,
        stages: [
          "choose Codex",
          ...(fixture
            ? ["pending code", "copy", "cancel", "second login"]
            : []),
          "connected",
          "Next enabled",
          "Say hello ready",
          "Close without model request",
        ],
        persisted_agent_kind: settings.agent_settings.agent_kind,
        persisted_acp_server: settings.agent_settings.acp_server,
        entered_api_key: false,
        model_requests: 0,
        page_errors: errors,
        http_observations: requests,
        credential_reuse_between_llm_and_acp:
          "Not implemented; separate proposal",
      },
      null,
      2,
    ) + "\n",
  );
  video = await page.video().path();
} catch (error) {
  await page.screenshot({ path: resolve(scratch, "failure.png") });
  await writeFile(
    resolve(scratch, "failure-state.txt"),
    await page.locator("body").innerText(),
  );
  throw error;
} finally {
  await context.close();
  await browser.close();
}
await copyFile(video, resolve(artifacts, prefix + ".webm"));
console.log(
  prefix +
    ": passed; first-run onboarding reached ready with no API key/model request.",
);
