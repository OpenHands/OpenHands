#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

const SUBSCRIPTION_BASE_URL = "https://chatgpt.com/backend-api/codex";
const SUBSCRIPTION_PROFILE = "issue-1595-subscription";
const API_PROFILE = "issue-1595-api-key";
const DEFAULT_AGENT_PROFILE = "default";
const SUCCESS_TOKEN = "ISSUE1595_PR_SUCCESS";
const NON_SUB_TOKEN = "ISSUE1595_API_KEY_SUCCESS";
const MISSING_SCOPE_TEXT = "Missing scopes: api.responses.write";

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = "true";
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}

const args = parseArgs(process.argv.slice(2));
const label = args.label ?? "run";
const mode = args.mode ?? "subscription";
const expected = args.expected ?? "unknown";
const baseUrl = (args.baseUrl ?? "").replace(/\/$/, "");
const outDir = args.outDir ?? path.join(".pr", "issue-1595", "runs", label);
const commitSha = args.sha ?? "unknown";
const sessionApiKey =
  process.env.ISSUE_1595_SESSION_API_KEY ??
  process.env.LOCAL_BACKEND_API_KEY ??
  process.env.SESSION_API_KEY ??
  process.env.OH_SESSION_API_KEYS_0 ??
  "";

if (!baseUrl) throw new Error("--baseUrl is required");
if (!sessionApiKey) throw new Error("A session API key env var is required");

const sensitiveKeyPattern =
  /api[_-]?key|authorization|cookie|secret|token|device_code|user_code|email|account|session|conversation_id|app_conversation_id|task_id|id$/i;
const publicHashKeys = new Set(["commitSha", "build_git_sha"]);
const jwtPattern = /\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\b/g;
const uuidPattern =
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;
const longHexPattern = /\b[0-9a-f]{40,}\b/gi;
const fernetPattern = /\bgAAAAA[A-Za-z0-9_-]{20,}\b/g;
const displayedApiKeyPattern =
  /\b(?:Received\s+)?API Key\s*=\s*[^,\n\r}]+/gi;
const abbreviatedOpenAiKeyPattern = /\bsk-[A-Za-z0-9._-]*\.{3}[A-Za-z0-9._-]*\b/g;
const conversationTitlePattern = /\bConversation [A-Za-z0-9_-]{4,}\b/g;

function redactString(value) {
  return value
    .replace(displayedApiKeyPattern, "API Key = <redacted>")
    .replace(abbreviatedOpenAiKeyPattern, "<redacted>")
    .replace(conversationTitlePattern, "Conversation <redacted>")
    .replace(jwtPattern, "<jwt-redacted>")
    .replace(fernetPattern, "<encrypted-secret-redacted>")
    .replace(uuidPattern, "<uuid-redacted>")
    .replace(longHexPattern, "<hex-redacted>");
}

function sanitize(value, key = "") {
  if (value == null) return value;
  if (typeof value === "string") {
    if (publicHashKeys.has(key)) return value;
    if (sensitiveKeyPattern.test(key)) return "<redacted>";
    return redactString(value);
  }
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) return value.map((item) => sanitize(item, key));
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([entryKey, entryValue]) => [
        entryKey,
        sanitize(entryValue, entryKey),
      ]),
    );
  }
  return String(value);
}

function isRetryableDomRace(message) {
  return (
    message.includes("Execution context was destroyed") ||
    message.includes("Element is not attached") ||
    message.includes("element was detached")
  );
}

async function writeJson(fileName, value) {
  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(
    path.join(outDir, fileName),
    `${JSON.stringify(sanitize(value), null, 2)}\n`,
  );
}

async function apiFetch(apiPath, init = {}) {
  const headers = {
    Accept: "application/json",
    "X-Session-API-Key": sessionApiKey,
    ...(init.headers ?? {}),
  };
  let body = init.body;
  if (init.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(init.json);
  }
  const response = await fetch(`${baseUrl}${apiPath}`, {
    ...init,
    headers,
    body,
  });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { text };
  }
  if (!response.ok) {
    const error = new Error(`${init.method ?? "GET"} ${apiPath} -> ${response.status}`);
    error.response = { status: response.status, data: sanitize(data) };
    throw error;
  }
  return data;
}

async function waitForStack() {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < 120_000) {
    try {
      await apiFetch("/server_info");
      await apiFetch("/api/settings");
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 1_000));
    }
  }
  throw lastError ?? new Error("Timed out waiting for stack readiness");
}

async function chooseSubscriptionModel() {
  const response = await apiFetch("/api/llm/subscription/openai/models");
  const models = Array.isArray(response?.models) ? response.models : [];
  return (
    models.find((model) => model === "gpt-5.5") ??
    models.find((model) => model.includes("gpt-5.5")) ??
    models.find((model) => model.includes("codex")) ??
    models[0] ??
    "gpt-5.5"
  );
}

async function saveLlmProfile(name, llm) {
  await apiFetch(`/api/profiles/${encodeURIComponent(name)}`, {
    method: "POST",
    json: {
      llm,
      include_secrets: true,
    },
  });
  await apiFetch(`/api/profiles/${encodeURIComponent(name)}/activate`, {
    method: "POST",
    json: {},
  });
}

async function saveDefaultAgentProfile(llmProfileRef) {
  await apiFetch(`/api/agent-profiles/${encodeURIComponent(DEFAULT_AGENT_PROFILE)}`, {
    method: "POST",
    json: {
      agent_kind: "openhands",
      llm_profile_ref: llmProfileRef,
    },
  });
  const detail = await apiFetch(
    `/api/agent-profiles/${encodeURIComponent(DEFAULT_AGENT_PROFILE)}`,
  );
  const id = detail?.profile?.id;
  if (typeof id !== "string" || id.length === 0) {
    throw new Error("Default agent profile response did not include an id");
  }
  await apiFetch(`/api/agent-profiles/${encodeURIComponent(id)}/activate`, {
    method: "POST",
    json: {},
  });
  return detail;
}

async function configureSubscription() {
  const model = args.model ?? (await chooseSubscriptionModel());
  const llm = {
    model,
    api_key: "stale-api-key-that-must-be-stripped",
    base_url: SUBSCRIPTION_BASE_URL,
    auth_type: "subscription",
    subscription_vendor: "openai",
  };
  await apiFetch("/api/settings", {
    method: "PATCH",
    json: {
      agent_settings_diff: {
        llm,
        condenser: { enabled: false },
      },
      conversation_settings_diff: {
        confirmation_mode: false,
        max_iterations: 4,
      },
    },
  });
  await saveLlmProfile(SUBSCRIPTION_PROFILE, llm);
  const agentProfile = await saveDefaultAgentProfile(SUBSCRIPTION_PROFILE);
  const status = await apiFetch("/api/llm/subscription/openai/status");
  const settings = await apiFetch("/api/settings", {
    headers: { "X-Expose-Secrets": "encrypted" },
  });
  const profile = await apiFetch(`/api/profiles/${encodeURIComponent(SUBSCRIPTION_PROFILE)}`, {
    headers: { "X-Expose-Secrets": "encrypted" },
  });
  return { model, llm, status, settings, profile, agentProfile };
}

async function configureApiKey() {
  const model = args.model ?? process.env.ISSUE_1595_NON_SUB_MODEL ?? process.env.LLM_MODEL;
  const apiKey = process.env.ISSUE_1595_NON_SUB_API_KEY ?? process.env.LLM_API_KEY;
  const apiBaseUrl =
    args.apiBaseUrl ??
    process.env.ISSUE_1595_NON_SUB_BASE_URL ??
    "https://llm-proxy.app.all-hands.dev";
  if (!model || !apiKey) {
    throw new Error("Non-subscription run requires model and API key env vars");
  }
  const llm = {
    model,
    api_key: apiKey,
    base_url: apiBaseUrl,
    auth_type: "api_key",
  };
  await apiFetch("/api/settings", {
    method: "PATCH",
    json: {
      agent_settings_diff: {
        llm,
        condenser: { enabled: false },
      },
      conversation_settings_diff: {
        confirmation_mode: false,
        max_iterations: 4,
      },
    },
  });
  await saveLlmProfile(API_PROFILE, llm);
  const agentProfile = await saveDefaultAgentProfile(API_PROFILE);
  const settings = await apiFetch("/api/settings", {
    headers: { "X-Expose-Secrets": "encrypted" },
  });
  const profile = await apiFetch(`/api/profiles/${encodeURIComponent(API_PROFILE)}`, {
    headers: { "X-Expose-Secrets": "encrypted" },
  });
  return { model, llm, settings, profile, agentProfile };
}

async function setChatInput(page, text) {
  const input = page.getByTestId("chat-input");
  for (let attempt = 0; attempt < 5; attempt += 1) {
    await input.waitFor({ timeout: 60_000 });
    try {
      await input.evaluate((el, inputText) => {
        el.focus();
        el.textContent = inputText;
        el.dispatchEvent(
          new InputEvent("input", {
            bubbles: true,
            data: inputText,
            inputType: "insertText",
          }),
        );
      }, text);
      return;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!isRetryableDomRace(message) || attempt === 4) {
        throw error;
      }
      await page.waitForTimeout(1_000);
    }
  }
}

async function clickSubmitButton(page) {
  const button = page.getByTestId("submit-button");
  for (let attempt = 0; attempt < 5; attempt += 1) {
    await button.waitFor({ timeout: 60_000 });
    try {
      await button.click({ force: true, noWaitAfter: true, timeout: 5_000 });
      return;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!isRetryableDomRace(message) || attempt === 4) {
        throw error;
      }
      await page.waitForTimeout(1_000);
    }
  }
}

async function installLabel(page, text) {
  try {
    await page.evaluate((labelText) => {
      const existing = document.querySelector("[data-issue-1595-label]");
      existing?.remove();
      const el = document.createElement("div");
      el.dataset.issue1595Label = "true";
      el.textContent = labelText;
      Object.assign(el.style, {
        position: "fixed",
        top: "10px",
        left: "10px",
        zIndex: "2147483647",
        padding: "8px 10px",
        background: "#111827",
        color: "#f9fafb",
        font: "600 13px system-ui, sans-serif",
        border: "1px solid #6b7280",
        borderRadius: "6px",
        boxShadow: "0 4px 20px rgba(0,0,0,.35)",
        pointerEvents: "none",
      });
      document.body.appendChild(el);
    }, text);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!isRetryableDomRace(message)) {
      throw error;
    }
  }
}

async function collectConversation(conversationId) {
  const conversation = await apiFetch(`/api/conversations/${encodeURIComponent(conversationId)}`);
  const events = await apiFetch(
    `/api/conversations/${encodeURIComponent(conversationId)}/events/search?limit=80`,
  );
  const eventItems = Array.isArray(events?.events)
    ? events.events
    : Array.isArray(events?.items)
      ? events.items
      : [];
  return {
    conversation: {
      execution_status: conversation?.execution_status,
      current_model_id: conversation?.current_model_id,
      current_model_name: conversation?.current_model_name,
      workspace_kind: conversation?.workspace?.kind,
    },
    eventEvidence: eventItems.map(summarizeEventEvidence).filter(Boolean),
  };
}

function excerptAround(text, marker) {
  const index = text.indexOf(marker);
  if (index === -1) return null;
  const start = Math.max(0, index - 180);
  const end = Math.min(text.length, index + marker.length + 180);
  return text.slice(start, end);
}

function summarizeEventEvidence(event) {
  const sanitizedEvent = sanitize(event);
  const text = JSON.stringify(sanitizedEvent);
  const markers = [MISSING_SCOPE_TEXT, SUCCESS_TOKEN, NON_SUB_TOKEN].filter((marker) =>
    text.includes(marker),
  );
  const kind = event?.kind ?? event?.type ?? event?.source;
  if (markers.length === 0 && !kind) return null;
  return {
    kind,
    source: event?.source,
    llm_role: event?.llm_message?.role,
    markers,
    excerpt: markers.length > 0 ? excerptAround(text, markers[0]) : null,
  };
}

function summarizeConversationRequest(payload) {
  if (!payload || typeof payload !== "object") return payload;
  const agentSettings = payload.agent_settings ?? {};
  const llm = agentSettings.llm ?? null;
  return {
    agent_settings: {
      agent_kind: agentSettings.agent_kind,
      agent: agentSettings.agent,
      llm,
      tools: Array.isArray(agentSettings.tools)
        ? agentSettings.tools.map((tool) => ({ name: tool?.name }))
        : agentSettings.tools,
      runtime_services_suffix_present: typeof agentSettings.agent_context?.system_message_suffix === "string",
    },
    workspace: payload.workspace
      ? { kind: payload.workspace.kind, working_dir: "<redacted>" }
      : undefined,
    confirmation_policy: payload.confirmation_policy,
    max_iterations: payload.max_iterations,
    stuck_detection: payload.stuck_detection,
    worktree: payload.worktree,
    secrets_encrypted: payload.secrets_encrypted,
    initial_message_run: payload.initial_message?.run,
    secret_names: payload.secrets ? Object.keys(payload.secrets).sort() : [],
  };
}

async function runBrowserFlow(config) {
  const videoDir = path.join(outDir, "video");
  await fs.mkdir(videoDir, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    recordVideo: { dir: videoDir, size: { width: 1280, height: 900 } },
  });
  await context.addInitScript(
    ({ apiKey }) => {
      window.localStorage.setItem("analytics-consent", "false");
      window.localStorage.setItem("openhands-telemetry-consent", "denied");
      window.localStorage.setItem("openhands-telemetry-first-use", "true");
      window.localStorage.setItem("openhands-onboarded", "1");
      window.localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "default-local",
            name: "Local",
            host: window.location.origin,
            apiKey,
            kind: "local",
          },
        ]),
      );
    },
    { apiKey: sessionApiKey },
  );
  await context.route(/posthog\.com|z\.openhands\.dev/, (route) =>
    route.abort("blockedbyclient"),
  );
  const page = await context.newPage();
  const consoleMessages = [];
  const pageErrors = [];
  const clientRequests = [];
  let capturedConversationPayload = null;

  page.on("console", (message) => {
    consoleMessages.push({
      type: message.type(),
      text: sanitize(message.text()),
    });
  });
  page.on("pageerror", (error) => {
    pageErrors.push(sanitize(error.stack ?? error.message));
  });
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname === "/api/conversations" && request.method() === "POST") {
      try {
        capturedConversationPayload = summarizeConversationRequest(request.postDataJSON());
      } catch {
        capturedConversationPayload = { parse_error: true };
      }
    }
    if (
      url.pathname.includes("/api/llm/subscription/") ||
      url.pathname === "/api/conversations" ||
      url.pathname === "/api/settings"
    ) {
      clientRequests.push({
        method: request.method(),
        path: url.pathname,
        postDataSummary:
          request.method() === "POST" && url.pathname === "/api/conversations"
            ? capturedConversationPayload
            : undefined,
      });
    }
  });

  let conversationId = null;
  let outcome = "unknown";
  let pageText = "";
  let apiConversation = null;
  let missingScopeVisible = false;
  let agentReplyVisible = false;
  const overlay =
    mode === "subscription"
      ? `${label} ${commitSha.slice(0, 8)} | subscription ${expected}`
      : `${label} ${commitSha.slice(0, 8)} | api-key non-subscription`;

  try {
    await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded" });
    await page.getByTestId("home-chat-launcher").waitFor({ timeout: 60_000 });
    await page.waitForTimeout(1_000);
    await page.getByTestId("home-chat-launcher").waitFor({ timeout: 60_000 });
    await installLabel(page, overlay);
    await page.screenshot({ path: path.join(outDir, "01-home.png") });

    const prompt =
      mode === "subscription"
        ? `Reply exactly with ${SUCCESS_TOKEN} and do not use tools.`
        : `Reply exactly with ${NON_SUB_TOKEN} and do not use tools.`;
    await setChatInput(page, prompt);
    await clickSubmitButton(page);
    await page.waitForURL(/\/conversations\/[^/?#]+/, { timeout: 60_000 });
    await installLabel(page, overlay);
    const match = page.url().match(/\/conversations\/([^/?#]+)/);
    conversationId = match?.[1] ? decodeURIComponent(match[1]) : null;
    await page.screenshot({ path: path.join(outDir, "02-conversation-started.png") });

    if (expected === "missing-scopes") {
      await page.getByText(MISSING_SCOPE_TEXT).first().waitFor({ timeout: 180_000 });
      missingScopeVisible = true;
      outcome = "missing-scopes-visible";
    } else {
      const token = mode === "subscription" ? SUCCESS_TOKEN : NON_SUB_TOKEN;
      await page.getByTestId("agent-message").filter({ hasText: token }).first().waitFor({
        timeout: 180_000,
      });
      agentReplyVisible = true;
      outcome = "agent-reply-visible";
    }
    await installLabel(page, `${overlay} | ${outcome}`);
    await page.screenshot({ path: path.join(outDir, "03-final.png") });
  } catch (error) {
    outcome = `error: ${error instanceof Error ? error.message : String(error)}`;
    try {
      await installLabel(page, `${overlay} | ${outcome.slice(0, 90)}`);
      await page.screenshot({ path: path.join(outDir, "03-final-error.png") });
    } catch {
      // best-effort screenshot
    }
  } finally {
    pageText = sanitize(await page.locator("body").textContent().catch(() => ""));
    if (conversationId) {
      try {
        apiConversation = await collectConversation(conversationId);
      } catch (error) {
        apiConversation = {
          collect_error: error instanceof Error ? error.message : String(error),
          response: error?.response,
        };
      }
    }
    await context.close();
    await browser.close();
  }

  const videos = await fs.readdir(videoDir).catch(() => []);
  return {
    outcome,
    conversationId: conversationId ? "<conversation-id-redacted>" : null,
    capturedConversationPayload,
    clientRequests,
    consoleMessages,
    pageErrors,
    pageText,
    pageAssertions: {
      missingScopeVisible,
      agentReplyVisible,
    },
    apiConversation,
    videos: videos.map((file) => path.join("video", file)),
  };
}

async function main() {
  await fs.mkdir(outDir, { recursive: true });
  await waitForStack();
  const serverInfo = await apiFetch("/server_info");
  const configuration =
    mode === "subscription"
      ? await configureSubscription()
      : await configureApiKey();
  const browser = await runBrowserFlow(configuration);
  const result = {
    label,
    mode,
    expected,
    commitSha,
    baseUrl,
    serverInfo,
    configuration,
    browser,
    generatedAt: new Date().toISOString(),
  };
  await writeJson("result.json", result);
  console.log(
    JSON.stringify({
      label,
      mode,
      expected,
      outcome: browser.outcome,
      result: path.join(outDir, "result.json"),
    }),
  );
  if (expected === "missing-scopes" && browser.outcome !== "missing-scopes-visible") {
    process.exitCode = 2;
  }
  if (expected === "success" && browser.outcome !== "agent-reply-visible") {
    process.exitCode = 2;
  }
}

await main();
