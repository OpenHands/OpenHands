import { randomUUID } from "node:crypto";
import {
  expect,
  type APIRequestContext,
  type Locator,
  type Page,
} from "@playwright/test";
import { CANVAS_UI_CLIENT_TOOL } from "../../../../src/api/canvas-ui-client-tool";

export const BACKEND_URL =
  process.env.LIVE_E2E_BACKEND_URL ?? "http://127.0.0.1:18100";
export const EXPECTED_BASH_OUTPUT_TOKEN = "LIVE_AGENT_CANVAS_E2E_BASH_OK";
export const EXPECTED_BASH_COMMAND = `printf '${EXPECTED_BASH_OUTPUT_TOKEN}\\n'`;
export const EXPECTED_REPLY_TOKEN = "LIVE_AGENT_CANVAS_E2E_OK";
const POSTHOG_URL_PATTERN =
  /^https?:\/\/(?:(?:[^/]+\.)*posthog\.com|z\.openhands\.dev)(?:\/|$)/;

function firstNonEmpty(...values: Array<string | undefined>) {
  return values.find((value) => value?.trim()) ?? "";
}

const liveLLMApiKey = process.env.LIVE_E2E_LLM_API_KEY;
const openAIKey = process.env.OPENAI_API_KEY;
const anthropicKey = process.env.ANTHROPIC_API_KEY;
const proxyLLMKey = process.env.LLM_API_KEY;
const llmApiKey = firstNonEmpty(
  liveLLMApiKey,
  openAIKey,
  anthropicKey,
  proxyLLMKey,
);
const usesProxyKey = Boolean(
  liveLLMApiKey?.trim() ||
  (!openAIKey?.trim() && !anthropicKey?.trim() && proxyLLMKey?.trim()),
);
const llmBaseUrl =
  process.env.LIVE_E2E_LLM_BASE_URL ??
  (usesProxyKey ? "https://llm-proxy.app.all-hands.dev" : "");
const llmModel =
  process.env.LIVE_E2E_LLM_MODEL ??
  (llmBaseUrl
    ? "openhands/claude-haiku-4-5-20251001"
    : openAIKey?.trim()
      ? "openai/gpt-5.4-mini"
      : "anthropic/claude-haiku-4-5-20251001");
const DEFAULT_AGENT_TOOLS = [
  { name: "terminal", params: {} },
  { name: "file_editor", params: {} },
  { name: "task_tracker", params: {} },
];
export const sessionApiKey = firstNonEmpty(
  process.env.LIVE_E2E_SESSION_API_KEY,
  process.env.LOCAL_BACKEND_API_KEY,
);

if (!sessionApiKey) {
  throw new Error("LIVE_E2E_SESSION_API_KEY must be set for live E2E.");
}

const LIVE_LLM_PROFILE_NAME = `live-e2e-${sessionApiKey.slice(0, 8)}`;
export const hasLiveLLMConfig = Boolean(llmApiKey);
export const missingLiveLLMConfigMessage =
  "Set LIVE_E2E_LLM_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or LLM_API_KEY to run live E2E.";

interface ConfigureLiveAgentServerOptions {
  enableCritic?: boolean;
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export async function routeBackendSessionApiKey(page: Page) {
  const backendOrigin = new URL(BACKEND_URL).origin;
  await page.route(
    new RegExp(`^${escapeRegExp(backendOrigin)}(?:/|$)`),
    async (route) => {
      await route.continue({
        headers: {
          ...route.request().headers(),
          "X-Session-API-Key": sessionApiKey,
        },
      });
    },
  );
}

export function getLiveArtifactMask(page: Page): Locator[] {
  return [
    page.locator('input[type="password"]'),
    page.locator('[data-sensitive="true"]'),
    page.locator('[data-testid*="secret" i]'),
    page.locator('[data-testid*="token" i]'),
    page.locator('[data-testid*="api-key" i]'),
    page.getByText(
      /(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|LIVE_E2E_LLM_API_KEY|LLM_API_KEY|LOCAL_BACKEND_API_KEY|SESSION_API_KEY|X-Session-API-Key)\s*[:=]\s*\S+/i,
    ),
    page.getByText(
      /(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|(?:api[_ -]?key|secret|password|token)\s*[:=]\s*\S+)/i,
    ),
  ];
}

export async function configureLiveAgentServer(
  request: APIRequestContext,
  options: ConfigureLiveAgentServerOptions = {},
) {
  if (!llmApiKey.trim()) {
    throw new Error(missingLiveLLMConfigMessage);
  }

  await ensureLiveLlmProfile(request);

  const settingsResponse = await request.patch(`${BACKEND_URL}/api/settings`, {
    headers: {
      "X-Session-API-Key": sessionApiKey,
    },
    data: {
      agent_settings_diff: {
        llm: buildLiveLlmSettings(),
        condenser: {
          enabled: false,
        },
        verification: buildLiveVerificationSettings(options),
      },
      conversation_settings_diff: {
        confirmation_mode: false,
        max_iterations: 6,
      },
    },
  });
  expect(
    settingsResponse.ok(),
    `PATCH /api/settings failed with ${settingsResponse.status()}; response body omitted because live LLM credentials are configured in this request.`,
  ).toBeTruthy();
}

async function ensureLiveLlmProfile(request: APIRequestContext) {
  const profileUrl = `${BACKEND_URL}/api/profiles/${encodeURIComponent(LIVE_LLM_PROFILE_NAME)}`;
  const headers = {
    "X-Session-API-Key": sessionApiKey,
  };

  await deleteLiveLlmProfile(request);

  const saveResponse = await request.post(profileUrl, {
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    data: {
      llm: buildLiveLlmSettings(),
      include_secrets: true,
    },
  });
  expect(
    saveResponse.ok(),
    `POST /api/profiles/${LIVE_LLM_PROFILE_NAME} failed with ${saveResponse.status()}; response body omitted because live LLM credentials are configured in this request.`,
  ).toBeTruthy();

  const activateResponse = await request.post(`${profileUrl}/activate`, {
    headers,
  });
  expect(
    activateResponse.ok(),
    `POST /api/profiles/${LIVE_LLM_PROFILE_NAME}/activate failed with ${activateResponse.status()}; response body omitted because live LLM credentials are configured in this request.`,
  ).toBeTruthy();
}

export async function deleteLiveLlmProfile(request: APIRequestContext) {
  await request.delete(
    `${BACKEND_URL}/api/profiles/${encodeURIComponent(LIVE_LLM_PROFILE_NAME)}`,
    {
      headers: {
        "X-Session-API-Key": sessionApiKey,
      },
    },
  );
}

function buildLiveLlmSettings(): Record<string, string | number> {
  const llmSettings: Record<string, string | number> = {
    model: llmModel,
    api_key: llmApiKey,
    extended_thinking_budget: 1024,
    max_output_tokens: 2048,
    temperature: 0,
  };
  if (llmBaseUrl) {
    llmSettings.base_url = llmBaseUrl;
  }
  return llmSettings;
}

function buildLiveVerificationSettings(
  options: ConfigureLiveAgentServerOptions = {},
): Record<string, string | number | boolean> {
  const verificationSettings: Record<string, string | number | boolean> =
    options.enableCritic
      ? {
          critic_enabled: true,
          critic_mode: "finish_and_message",
          enable_iterative_refinement: false,
          critic_api_key: llmApiKey,
          critic_model_name: llmModel,
        }
      : {
          critic_enabled: false,
          enable_iterative_refinement: false,
        };
  if (options.enableCritic && llmBaseUrl) {
    verificationSettings.critic_server_url = llmBaseUrl;
  }
  return verificationSettings;
}

export async function createLiveConversation(
  request: APIRequestContext,
  options: ConfigureLiveAgentServerOptions = {},
) {
  if (!llmApiKey.trim()) {
    throw new Error(missingLiveLLMConfigMessage);
  }

  const conversationId = randomUUID();
  const response = await request.post(`${BACKEND_URL}/api/conversations`, {
    headers: {
      "X-Session-API-Key": sessionApiKey,
    },
    data: {
      conversation_id: conversationId,
      workspace: {
        kind: "LocalWorkspace",
        working_dir: `workspace/project/${conversationId}`,
      },
      worktree: true,
      max_iterations: 6,
      stuck_detection: true,
      autotitle: true,
      confirmation_policy: {
        kind: "NeverConfirm",
      },
      agent_settings: {
        agent_kind: "openhands",
        llm: buildLiveLlmSettings(),
        condenser: {
          enabled: false,
        },
        verification: buildLiveVerificationSettings(options),
        tools: DEFAULT_AGENT_TOOLS,
      },
      client_tools: [CANVAS_UI_CLIENT_TOOL],
    },
  });

  expect(
    response.ok(),
    `POST /api/conversations failed with ${response.status()}; response body omitted because live LLM credentials are configured in this request.`,
  ).toBeTruthy();

  const body = (await response.json()) as { id?: unknown };
  expect(
    typeof body.id === "string" && body.id.length > 0,
    "POST /api/conversations did not return a conversation id.",
  ).toBe(true);
  return body.id as string;
}

export async function enableLiveE2EFlags(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem("analytics-consent", "false");
    window.localStorage.setItem("openhands-telemetry-consent", "denied");
    window.localStorage.setItem("openhands-telemetry-first-use", "true");
    window.localStorage.setItem("openhands-onboarded", "1");
    window.localStorage.setItem("FEATURE_AUTOMATIONS", "true");
  });
}

export async function guardAgainstPostHogRequests(page: Page) {
  const postHogRequests: string[] = [];

  await page.route(POSTHOG_URL_PATTERN, async (route) => {
    postHogRequests.push(route.request().url());
    await route.fulfill({ status: 204, body: "" });
  });

  return {
    expectNoRequests() {
      expect(
        postHogRequests,
        [
          "Live E2E must not send analytics to PostHog.",
          "Keep VITE_DO_NOT_TRACK=1 and the live-test storage opt-out in place.",
        ].join(" "),
      ).toEqual([]);
    },
  };
}

export async function waitForPath(
  page: Page,
  pattern: RegExp,
  timeout = 60_000,
) {
  await expect
    .poll(
      async () => page.evaluate(() => window.location.pathname).catch(() => ""),
      { timeout },
    )
    .toMatch(pattern);
}

export async function openCreatedConversation(page: Page) {
  const conversationPathPattern = /\/conversations\/.+/;

  try {
    await waitForPath(page, conversationPathPattern, 10_000);
    return;
  } catch {
    await page.locator('a[href^="/conversations/"]').first().click();
    await waitForPath(page, conversationPathPattern);
  }
}

export async function waitForTestId(
  page: Page,
  testId: string,
  timeout = 60_000,
) {
  await expect
    .poll(
      async () =>
        page
          .evaluate(
            (testId) =>
              document.querySelector(`[data-testid="${testId}"]`) != null,
            testId,
          )
          .catch(() => false),
      { timeout },
    )
    .toBe(true);
}

export async function dismissAnalyticsModal(page: Page) {
  await page.waitForLoadState("domcontentloaded");

  await expect
    .poll(
      async () =>
        page
          .evaluate(() => {
            const hasAnalyticsDialog = Array.from(
              document.querySelectorAll('[role="dialog"]'),
            ).some((dialog) =>
              dialog.textContent?.includes("Help improve OpenHands"),
            );
            if (!hasAnalyticsDialog) {
              return true;
            }

            const confirmButton = Array.from(
              document.querySelectorAll("button"),
            ).find(
              (button) => button.textContent?.trim() === "Confirm preferences",
            );
            if (confirmButton instanceof HTMLButtonElement) {
              confirmButton.click();
            }
            return false;
          })
          .catch(() => false),
      { timeout: 5_000 },
    )
    .toBe(true);
}

export async function clickButtonByTestId(page: Page, testId: string) {
  await waitForTestId(page, testId);

  await page.evaluate((testId) => {
    const button = document.querySelector(`[data-testid="${testId}"]`);
    if (!(button instanceof HTMLButtonElement)) {
      throw new Error(`Button not found: ${testId}`);
    }
    button.click();
  }, testId);
}

export async function clickButtonByTestIdOrText(
  page: Page,
  testId: string,
  text: string,
) {
  await expect
    .poll(
      async () =>
        page.evaluate(
          ({ testId, text }) => {
            const byTestId = document.querySelector(
              `[data-testid="${testId}"]`,
            );
            if (byTestId instanceof HTMLButtonElement) {
              return true;
            }

            return Array.from(document.querySelectorAll("button")).some(
              (button) => button.textContent?.trim() === text,
            );
          },
          { testId, text },
        ),
      { timeout: 60_000 },
    )
    .toBe(true);

  await page.evaluate(
    ({ testId, text }) => {
      const byTestId = document.querySelector(`[data-testid="${testId}"]`);
      const button =
        byTestId instanceof HTMLButtonElement
          ? byTestId
          : Array.from(document.querySelectorAll("button")).find(
              (button) => button.textContent?.trim() === text,
            );
      if (!(button instanceof HTMLButtonElement)) {
        throw new Error(`Button not found: ${text}`);
      }
      button.click();
    },
    { testId, text },
  );
}

export async function fillChatInput(page: Page, text: string) {
  await waitForTestId(page, "chat-input");

  await page.evaluate((text) => {
    const input = document.querySelector('[data-testid="chat-input"]');
    if (!(input instanceof HTMLElement)) {
      throw new Error("Chat input not found");
    }
    input.focus();
    input.textContent = text;
    input.dispatchEvent(
      new InputEvent("input", {
        bubbles: true,
        data: text,
        inputType: "insertText",
      }),
    );
  }, text);
}

export async function waitForNonUserMessageText(page: Page, text: string) {
  await expect
    .poll(
      async () =>
        page
          .evaluate((text) => {
            const body = document.body.cloneNode(true);
            if (!(body instanceof HTMLElement)) {
              return false;
            }
            body
              .querySelectorAll('[data-testid="user-message"]')
              .forEach((node) => node.remove());
            return body.textContent?.includes(text) ?? false;
          }, text)
          .catch(() => false),
      { timeout: 120_000 },
    )
    .toBe(true);
}

export async function expandVisibleEventDetails(page: Page) {
  await expect
    .poll(
      async () =>
        page
          .evaluate(() => {
            const isVisible = (element: Element) => {
              const rect = element.getBoundingClientRect();
              return rect.width > 0 && rect.height > 0;
            };

            const buttons = Array.from(
              document.querySelectorAll('button[aria-label="Expand"]'),
            ).filter(isVisible);
            buttons.forEach((button) => {
              if (button instanceof HTMLButtonElement) {
                button.click();
              }
            });
            return buttons.length;
          })
          .catch(() => 0),
      { timeout: 5_000 },
    )
    .toBe(0);
}

export async function waitForAgentReply(page: Page) {
  await expect
    .poll(
      async () =>
        page
          .evaluate((expectedReplyToken) => {
            const hasReply = Array.from(
              document.querySelectorAll('[data-testid="agent-message"]'),
            ).some((element) =>
              element.textContent?.includes(expectedReplyToken),
            );
            if (hasReply) {
              return "reply";
            }
            if (document.body.textContent?.includes("Error occurred")) {
              return "error";
            }
            return "pending";
          }, EXPECTED_REPLY_TOKEN)
          .catch(() => "pending"),
      { timeout: 120_000 },
    )
    .toBe("reply");
}

export function getConversationIdFromURL(page: Page) {
  const match = page.url().match(/\/conversations\/([^/?#]+)/);
  expect(
    match?.[1],
    `Could not read conversation id from ${page.url()}`,
  ).toBeTruthy();
  return decodeURIComponent(match![1]);
}

export function getOptionalConversationIdFromURL(page: Page) {
  const match = page.url().match(/\/conversations\/([^/?#]+)/);
  return match?.[1] ? decodeURIComponent(match[1]) : "";
}

function eventTextContent(event: unknown) {
  if (!event || typeof event !== "object") {
    return "";
  }

  const content = (event as { observation?: { content?: unknown } }).observation
    ?.content;
  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((item) => {
      if (!item || typeof item !== "object") {
        return "";
      }
      const text = (item as { text?: unknown }).text;
      return typeof text === "string" ? text : "";
    })
    .join("\n");
}

function isSuccessfulBashObservation(event: unknown) {
  if (!event || typeof event !== "object") {
    return false;
  }

  const observation = (event as { observation?: Record<string, unknown> })
    .observation;
  if (!observation) {
    return false;
  }

  const kind = observation.kind;
  const isTerminalObservation =
    kind === "ExecuteBashObservation" || kind === "TerminalObservation";
  if (!isTerminalObservation) {
    return false;
  }

  const command =
    typeof observation.command === "string" ? observation.command : "";
  const output = eventTextContent(event);
  const exitCode = observation.exit_code;
  const failed =
    observation.error === true ||
    observation.is_error === true ||
    observation.timeout === true;

  return (
    command.includes(EXPECTED_BASH_OUTPUT_TOKEN) &&
    output.includes(EXPECTED_BASH_OUTPUT_TOKEN) &&
    exitCode === 0 &&
    !failed
  );
}

function hasCriticResult(event: unknown) {
  if (!event || typeof event !== "object") {
    return false;
  }

  const criticResult = (event as { critic_result?: unknown }).critic_result;
  if (!criticResult || typeof criticResult !== "object") {
    return false;
  }

  const score = (criticResult as { score?: unknown }).score;
  return typeof score === "number" && Number.isFinite(score);
}

export async function waitForSuccessfulBashObservation(
  request: APIRequestContext,
  conversationId: string,
) {
  await expect
    .poll(
      async () => {
        const response = await request.get(
          `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}/events/search`,
          {
            headers: {
              "X-Session-API-Key": sessionApiKey,
            },
            params: {
              limit: "100",
              sort_order: "TIMESTAMP_DESC",
            },
          },
        );

        if (!response.ok()) {
          return false;
        }

        const body = (await response.json()) as { items?: unknown[] };
        return body.items?.some(isSuccessfulBashObservation) ?? false;
      },
      { timeout: 120_000 },
    )
    .toBe(true);
}

export async function waitForCriticResultEvent(
  request: APIRequestContext,
  conversationId: string,
) {
  await expect
    .poll(
      async () => {
        const response = await request.get(
          `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}/events/search`,
          {
            headers: {
              "X-Session-API-Key": sessionApiKey,
            },
            params: {
              limit: "100",
              sort_order: "TIMESTAMP_DESC",
            },
          },
        );

        if (!response.ok()) {
          return false;
        }

        const body = (await response.json()) as { items?: unknown[] };
        return body.items?.some(hasCriticResult) ?? false;
      },
      { timeout: 180_000 },
    )
    .toBe(true);
}

export async function waitForCriticResultDisplay(page: Page) {
  await expect
    .poll(
      async () =>
        page
          .evaluate(() =>
            document.body.textContent?.includes(
              "Critic: agent success likelihood",
            ),
          )
          .catch(() => false),
      { timeout: 120_000 },
    )
    .toBe(true);
}
