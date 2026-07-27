/**
 * Shared plumbing for the live ACP-in-Docker e2e scripts
 * (`acp-docker-e2e.mts` — request-builder path; `acp-docker-app-e2e.mts` —
 * full app-orchestrator path): the per-provider plans + host credential
 * collectors, the backend registration, and the HTTP/poll helpers. Keeping the
 * plans in one place means a model default or credential knob can't drift
 * between the two scripts.
 *
 * Credentials are read from the host and never printed: Codex
 * ~/.codex/auth.json, the Claude Code OAuth token from the macOS keychain, and
 * the gcloud ADC for Gemini Vertex.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";

export const BASE = process.env.ACP_E2E_BASE_URL ?? "http://localhost:8010";
export const POLL_TIMEOUT_MS = Number(
  process.env.ACP_E2E_TIMEOUT_MS ?? 180_000,
);

/**
 * Point the app's backend registry at the container, exactly as if the user
 * had added it in the backend selector. Everything downstream (SecretsService,
 * SettingsService, buildStartConversationRequest's LookupSecret auth headers)
 * resolves the host through this.
 */
export function registerDockerBackend(): void {
  setRegisteredBackends([
    {
      id: "acp-docker",
      name: "ACP Docker",
      host: BASE,
      apiKey: "",
      kind: "local",
    },
  ]);
  setActiveSelection({ backendId: "acp-docker", orgId: null });
}

export type ProviderId = "codex" | "claude" | "gemini";

export interface ProviderPlan {
  id: ProviderId;
  /** ACP registry key sent as acp_server. */
  acpServer: string;
  /** acp_model to send (a model the account/Vertex project supports). */
  model: string;
  expectedToken: string;
  /** Container-credential map (name -> value) or null when creds are missing. */
  collectSecrets: () => Record<string, string> | null;
  /**
   * Optional ``acp_session_mode`` override (env-driven). Canvas itself sends
   * none — the SDK then uses the provider's registry default. For Gemini that
   * default (``yolo``) makes gemini-cli ≥0.43 error on ``set_session_mode``
   * during headless init (an SDK/gemini-cli issue, not a credential one); set
   * ``ACP_E2E_GEMINI_SESSION_MODE=default`` to confirm the credential path
   * end-to-end past that blocker.
   */
  sessionMode?: string;
}

function readFileTrimmed(file: string): string | null {
  try {
    const value = readFileSync(file, "utf-8");
    return value.trim().length > 0 ? value : null;
  } catch {
    return null;
  }
}

function claudeOAuthToken(): string | null {
  // macOS keychain entry written by Claude Code on login. On other platforms
  // (no `security` binary) this throws into the catch → null → the runner
  // reports the provider as "SKIP — credentials not present on host".
  try {
    const raw = execFileSync(
      "security",
      ["find-generic-password", "-s", "Claude Code-credentials", "-w"],
      { encoding: "utf-8" },
    );
    const token = JSON.parse(raw)?.claudeAiOauth?.accessToken;
    return typeof token === "string" && token.length > 0 ? token : null;
  } catch {
    return null;
  }
}

function gcloudProject(): string | null {
  try {
    return (
      execFileSync("gcloud", ["config", "get-value", "project"], {
        encoding: "utf-8",
      }).trim() || null
    );
  } catch {
    return null;
  }
}

export const PROVIDER_PLANS: ProviderPlan[] = [
  {
    id: "codex",
    acpServer: "codex",
    model: process.env.ACP_E2E_CODEX_MODEL ?? "gpt-5.5/medium",
    expectedToken: "ACPOK-CODEX",
    collectSecrets: () => {
      const auth = readFileTrimmed(path.join(homedir(), ".codex", "auth.json"));
      return auth ? { CODEX_AUTH_JSON: auth } : null;
    },
  },
  {
    id: "claude",
    acpServer: "claude-code",
    model: process.env.ACP_E2E_CLAUDE_MODEL ?? "claude-opus-4-7",
    expectedToken: "ACPOK-CLAUDE",
    collectSecrets: () => {
      const token = claudeOAuthToken();
      // NB: deliberately NOT setting ANTHROPIC_BASE_URL — an inherited base URL
      // breaks the OAuth token's bearer auth.
      return token ? { CLAUDE_CODE_OAUTH_TOKEN: token } : null;
    },
  },
  {
    id: "gemini",
    acpServer: "gemini-cli",
    model: process.env.ACP_E2E_GEMINI_MODEL ?? "gemini-2.5-pro",
    expectedToken: "ACPOK-GEMINI",
    collectSecrets: () => {
      const adc = readFileTrimmed(
        path.join(
          homedir(),
          ".config",
          "gcloud",
          "application_default_credentials.json",
        ),
      );
      const project = process.env.GOOGLE_CLOUD_PROJECT ?? gcloudProject();
      if (!adc || !project) return null;
      return {
        GOOGLE_APPLICATION_CREDENTIALS_JSON: adc,
        GOOGLE_CLOUD_PROJECT: project,
        GOOGLE_CLOUD_LOCATION:
          process.env.GOOGLE_CLOUD_LOCATION ?? "us-central1",
        GOOGLE_GENAI_USE_VERTEXAI: "true",
      };
    },
    sessionMode: process.env.ACP_E2E_GEMINI_SESSION_MODE,
  },
];

export function getProviderPlan(id: string): ProviderPlan | undefined {
  return PROVIDER_PLANS.find((plan) => plan.id === id);
}

export async function postJson(url: string, body: unknown): Promise<any> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`POST ${url} -> ${res.status}: ${text.slice(0, 800)}`);
  }
  return text ? JSON.parse(text) : null;
}

export async function getJson(url: string): Promise<any> {
  const res = await fetch(url);
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`GET ${url} -> ${res.status}: ${text.slice(0, 400)}`);
  }
  return text ? JSON.parse(text) : null;
}

// Terminal states for a single-turn run. NB: "idle" is deliberately NOT here —
// a freshly-created conversation reports "idle" before the agent starts, so
// treating it as terminal bails out before the reply exists. Wait for the run
// to actually finish (or error/stuck).
const TERMINAL = new Set(["finished", "error", "stuck", "stopped"]);

/** Poll the conversation until a terminal state (or timeout); returns the last
 * observed ``execution_status``. */
export async function pollUntilTerminal(conversationId: string): Promise<string> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  let status = "";
  while (Date.now() < deadline) {
    const info = await getJson(`${BASE}/api/conversations/${conversationId}`);
    status = String(info.execution_status ?? "").toLowerCase();
    if (TERMINAL.has(status)) break;
    await new Promise((r) => setTimeout(r, 2500));
  }
  return status;
}

/** Fetch the agent's final response as a plain string. */
export async function fetchFinalReply(conversationId: string): Promise<string> {
  const final = await getJson(
    `${BASE}/api/conversations/${conversationId}/agent_final_response`,
  );
  return typeof final === "string"
    ? final
    : String(final?.response ?? final?.content ?? JSON.stringify(final));
}
