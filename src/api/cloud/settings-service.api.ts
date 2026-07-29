import {
  type Provider,
  type Settings,
  type SettingsValue,
} from "#/types/settings";
import { type AppPreferences } from "../settings-service/settings-service.api";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

/**
 * The cloud Settings response is mostly flat — top-level fields like
 * `llm_model`, `provider_tokens_set`, etc., rather than the nested
 * `{ agent_settings, conversation_settings }` shape the local agent-server
 * uses. We deliberately do NOT remap cloud fields into the local shape:
 * the GUI's `Settings` type already supports both layouts (it has flat
 * fields AND nested `agent_settings`/`conversation_settings`), and
 * cloud-aware hooks like `useUserProviders` read directly from the flat
 * `provider_tokens_set` field. So the cloud response is passed through as
 * a `Partial<Settings>`, with a small derivation step to also populate
 * the nested fields the local-mode settings UI consumes.
 */
type CloudSettingsResponse = {
  llm_model?: string;
  llm_base_url?: string;
  llm_api_key?: string | null;
  llm_api_key_set?: boolean;
  search_api_key_set?: boolean;
  agent?: string;
  confirmation_mode?: boolean;
  security_analyzer?: string | null;
  max_iterations?: number | null;
  enable_default_condenser?: boolean;
  condenser_max_size?: number | null;
  enable_proactive_conversation_starters?: boolean;
  enable_solvability_analysis?: boolean;
  enable_sound_notifications?: boolean;
  language?: string;
  email?: string;
  email_verified?: boolean;
  git_user_name?: string;
  git_user_email?: string;
  title_llm_profile?: string | null;
  user_consents_to_analytics?: boolean | null;
  is_new_user?: boolean;
  remote_runtime_resource_factor?: number | null;
  max_budget_per_task?: number | null;
  provider_tokens_set?: Partial<Record<Provider, string | null>>;
  mcp_config?: Record<string, SettingsValue>;
  disabled_skills?: string[];
  agent_settings?: Record<string, SettingsValue> | null;
  conversation_settings?: Record<string, SettingsValue> | null;
  agent_settings_schema?: unknown;
  conversation_settings_schema?: unknown;
  [key: string]: unknown;
};

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud settings call requires a cloud backend.");
  }
  return active;
}

/**
 * Build the nested `agent_settings` block from the cloud's flat fields.
 * Used so the local-mode settings page (which renders against the nested
 * shape) shows the right values when the active backend is cloud.
 */
function deriveAgentSettings(
  flat: CloudSettingsResponse,
): Record<string, SettingsValue> {
  if (flat.agent_settings && Object.keys(flat.agent_settings).length > 0) {
    return flat.agent_settings;
  }
  const agent: Record<string, SettingsValue> = {};
  const llm: Record<string, SettingsValue> = {};
  if (typeof flat.llm_model === "string") llm.model = flat.llm_model;
  if (typeof flat.llm_base_url === "string") llm.base_url = flat.llm_base_url;
  if (typeof flat.llm_api_key === "string") llm.api_key = flat.llm_api_key;
  if (Object.keys(llm).length > 0) agent.llm = llm;

  const condenser: Record<string, SettingsValue> = {};
  if (typeof flat.enable_default_condenser === "boolean") {
    condenser.enabled = flat.enable_default_condenser;
  }
  if (typeof flat.condenser_max_size === "number") {
    condenser.max_size = flat.condenser_max_size;
  }
  if (Object.keys(condenser).length > 0) agent.condenser = condenser;

  if (typeof flat.agent === "string") agent.agent = flat.agent;
  if (flat.mcp_config && Object.keys(flat.mcp_config).length > 0) {
    agent.mcp_config = flat.mcp_config;
  }
  return agent;
}

function deriveConversationSettings(
  flat: CloudSettingsResponse,
): Record<string, SettingsValue> {
  if (
    flat.conversation_settings &&
    Object.keys(flat.conversation_settings).length > 0
  ) {
    return flat.conversation_settings;
  }
  const out: Record<string, SettingsValue> = {};
  if (typeof flat.confirmation_mode === "boolean") {
    out.confirmation_mode = flat.confirmation_mode;
  }
  if (
    typeof flat.security_analyzer === "string" ||
    flat.security_analyzer === null
  ) {
    out.security_analyzer = flat.security_analyzer;
  }
  if (typeof flat.max_iterations === "number") {
    out.max_iterations = flat.max_iterations;
  }
  return out;
}

/**
 * Fetch the cloud settings and return them as a `Partial<Settings>`.
 *
 * Top-level fields like `provider_tokens_set` are preserved unchanged so
 * the existing `useUserProviders` → `useAppInstallations` →
 * `useGitRepositories` chain (which reads `settings.provider_tokens_set`)
 * fires correctly in cloud mode. Nested `agent_settings` /
 * `conversation_settings` are derived for the settings page.
 */
export async function fetchCloudSettings(): Promise<Partial<Settings>> {
  const backend = getActiveCloudBackend();
  const flat = await callCloudProxy<CloudSettingsResponse>({
    backend,
    method: "GET",
    path: "/api/v1/settings",
  });

  return {
    ...flat,
    agent_settings: deriveAgentSettings(flat),
    conversation_settings: deriveConversationSettings(flat),
    llm_api_key_set: !!flat.llm_api_key_set,
    search_api_key_set: !!flat.search_api_key_set,
    provider_tokens_set: flat.provider_tokens_set,
  } as Partial<Settings>;
}

export async function saveCloudSettings(diff: {
  agent_settings_diff?: Record<string, SettingsValue>;
  conversation_settings_diff?: Record<string, SettingsValue>;
  /**
   * App-level user preferences (language, sound notifications, disabled
   * skills, …). The cloud `POST /api/v1/settings` consumes these as flat
   * top-level fields, so this helper iterates the object and assigns each
   * key onto the request body.
   *
   * Note: `app_preferences.disabled_skills` is the canonical source for the
   * skill list since `AppPreferences` was unified in agent-server 1.27.
   * Callers that still pass `disabled_skills` separately should migrate to
   * setting it under `app_preferences` instead.
   */
  app_preferences?: AppPreferences;
}): Promise<void> {
  const backend = getActiveCloudBackend();
  const body: Record<string, unknown> = {};
  if (diff.agent_settings_diff) {
    const agentDiff: Record<string, SettingsValue> = {
      ...diff.agent_settings_diff,
    };
    // The cloud validates agent settings against the SDK's
    // OpenHandsAgentSettings, whose `agent_context` is a required
    // AgentContext (not Optional). A literal `agent_context: null` fails
    // backend validation, so drop it and let the backend keep/default it.
    // See OpenHands/agent-canvas#981.
    if (agentDiff.agent_context === null) {
      delete agentDiff.agent_context;
    }
    if (Object.keys(agentDiff).length > 0) {
      body.agent_settings_diff = agentDiff;
    }
  }
  if (
    diff.conversation_settings_diff &&
    Object.keys(diff.conversation_settings_diff).length > 0
  ) {
    body.conversation_settings_diff = diff.conversation_settings_diff;
  }
  // Flat top-level app-preference fields (language, git_user_name,
  // disabled_skills, …). The cloud POST /api/v1/settings stores these
  // directly; see `CloudSettingsResponse` and the MSW handler in
  // `src/mocks/settings-handlers.ts` for the accepted shape. `undefined`
  // values are skipped so the cloud server keeps the prior value; `null`
  // and empty arrays (e.g. re-enabling every skill) round-trip as
  // explicit clears.
  if (diff.app_preferences) {
    for (const [key, value] of Object.entries(diff.app_preferences)) {
      if (value !== undefined) {
        body[key] = value;
      }
    }
  }
  await callCloudProxy<unknown>({
    backend,
    method: "POST",
    path: "/api/v1/settings",
    body,
  });
}

export async function fetchCloudSettingsSchema(): Promise<unknown> {
  const backend = getActiveCloudBackend();
  return callCloudProxy<unknown>({
    backend,
    method: "GET",
    path: "/api/v1/settings/agent-schema",
  });
}

export async function fetchCloudConversationSettingsSchema(): Promise<unknown> {
  const backend = getActiveCloudBackend();
  return callCloudProxy<unknown>({
    backend,
    method: "GET",
    path: "/api/v1/settings/conversation-schema",
  });
}
