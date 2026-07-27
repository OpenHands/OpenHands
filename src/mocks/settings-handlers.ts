import { http, delay, HttpResponse } from "msw";
import { WebClientConfig } from "#/api/option-service/option.types";
import type { SaveProfileRequest } from "#/api/profiles-service/profiles-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings, SettingsValue } from "#/types/settings";
import {
  OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH,
  OPENAI_SUBSCRIPTION_DEVICE_START_PATH,
  OPENAI_SUBSCRIPTION_LOGOUT_PATH,
  OPENAI_SUBSCRIPTION_MODELS_PATH,
  OPENAI_SUBSCRIPTION_STATUS_PATH,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";

/** Simple recursive merge — objects merge, scalars overwrite. */
function deepMerge(
  base: Record<string, unknown>,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...base };
  for (const [key, value] of Object.entries(patch)) {
    if (
      value != null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      typeof result[key] === "object" &&
      result[key] != null &&
      !Array.isArray(result[key])
    ) {
      result[key] = deepMerge(
        result[key] as Record<string, unknown>,
        value as Record<string, unknown>,
      );
    } else {
      result[key] = value;
    }
  }
  return result;
}

const DEFAULT_AGENT_SETTINGS = DEFAULT_SETTINGS.agent_settings ?? {};
const llmDefaults = (DEFAULT_AGENT_SETTINGS as Record<string, unknown>).llm as
  | Record<string, unknown>
  | undefined;
const DEFAULT_MODEL =
  typeof llmDefaults?.model === "string"
    ? llmDefaults.model
    : "openhands/claude-opus-4-5-20251101";

export const createMockWebClientConfig = (
  overrides: Partial<WebClientConfig> = {},
): WebClientConfig => ({
  feature_flags: {
    hide_llm_settings: false,
    hide_users_page: false,
    ...overrides.feature_flags,
  },
  providers_configured: [],
  maintenance_start_time: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: new Date().toISOString(),
  ...overrides,
});

const MOCK_AGENT_SETTINGS_SCHEMA: NonNullable<
  Settings["agent_settings_schema"]
> = {
  model_name: "AgentSettings",
  sections: [
    {
      key: "general",
      label: "General",
      fields: [
        {
          key: "enable_sub_agents",
          label: "Enable sub-agents",
          description:
            "Allow the agent to delegate work to specialized built-in sub-agents.",
          section: "general",
          section_label: "General",
          value_type: "boolean",
          default: false,
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: false,
        },
        {
          key: "tool_concurrency_limit",
          label: "Parallel tool calls",
          description:
            "Maximum number of tool calls to execute concurrently per agent step. 1 = sequential (default).",
          section: "general",
          section_label: "General",
          value_type: "integer",
          default: 1,
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: false,
        },
      ],
    },
    {
      key: "llm",
      label: "LLM",
      fields: [
        {
          key: "llm.model",
          label: "Model",
          description: "Select the model to use for this conversation.",
          section: "llm",
          section_label: "LLM",
          value_type: "string",
          default: DEFAULT_MODEL,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: false,
          required: true,
        },
        {
          key: "llm.api_key",
          label: "API Key",
          description:
            "Provide the API key used to authenticate requests for the selected model.",

          section: "llm",
          section_label: "LLM",
          value_type: "string",
          default: null,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: true,
          required: false,
        },
        {
          key: "llm.base_url",
          description:
            "Override the model provider's default API base URL when needed.",

          label: "Base URL",
          section: "llm",
          section_label: "LLM",
          value_type: "string",
          default: null,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: false,
          required: false,
        },
        {
          key: "llm.temperature",
          label: "Temperature",
          description: "Adjust randomness for non-deterministic model outputs.",
          section: "llm",
          section_label: "LLM",
          value_type: "number",
          default: null,
          choices: [],
          depends_on: [],
          prominence: "minor",
          secret: false,
          required: false,
        },
      ],
    },
    {
      key: "verification",
      label: "Verification",
      fields: [
        {
          key: "verification.critic_enabled",
          label: "Enable Critic",
          description:
            "Enable an additional critic pass to review the agent's work.",
          section: "verification",
          section_label: "Verification",
          value_type: "boolean",
          default: false,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: false,
          required: false,
        },
        {
          key: "verification.critic_mode",
          label: "Critic Mode",
          description: "Choose when the critic should review and intervene.",
          section: "verification",
          section_label: "Verification",
          value_type: "string",
          default: "finish_and_message",
          choices: [
            {
              label: "finish_and_message",
              value: "finish_and_message",
            },
            { label: "all_actions", value: "all_actions" },
          ],
          depends_on: ["verification.critic_enabled"],
          prominence: "major",
          secret: false,
          required: false,
        },
        {
          key: "verification.enable_iterative_refinement",
          label: "Enable Iterative Refinement",
          description:
            "Let the critic send the agent back to refine its work when issues are found.",
          section: "verification",
          section_label: "Verification",
          value_type: "boolean",
          default: false,
          choices: [],
          depends_on: ["verification.critic_enabled"],
          prominence: "critical",
          secret: false,
          required: false,
        },
        // Rendered as a full-width row (see FIELD_FULL_WIDTH_KEYS) below the
        // two critical-prominence toggles so the input + OpenHands Cloud help
        // link have room to breathe.
        {
          key: "verification.critic_api_key",
          label: "Critic API Key",
          description:
            "If OpenHands is selected as your active LLM provider, leave this empty; the critic reuses the OpenHands Provider LLM Key.",
          section: "verification",
          section_label: "Verification",
          value_type: "string",
          default: null,
          choices: [],
          depends_on: ["verification.critic_enabled"],
          prominence: "critical",
          secret: true,
          required: false,
        },
        {
          key: "verification.critic_threshold",
          label: "Critic Threshold",
          description:
            "Critic success threshold used for iterative refinement.",
          section: "verification",
          section_label: "Verification",
          value_type: "number",
          default: 0.6,
          choices: [],
          depends_on: [
            "verification.critic_enabled",
            "verification.enable_iterative_refinement",
          ],
          prominence: "minor",
          secret: false,
          required: false,
        },
        {
          key: "verification.max_refinement_iterations",
          label: "Max Refinement Iterations",
          description:
            "Maximum number of refinement attempts after critic feedback.",
          section: "verification",
          section_label: "Verification",
          value_type: "integer",
          default: 3,
          choices: [],
          depends_on: [
            "verification.critic_enabled",
            "verification.enable_iterative_refinement",
          ],
          prominence: "minor",
          secret: false,
          required: false,
        },
        {
          key: "verification.critic_server_url",
          label: "Critic Server URL",
          description: "Override the critic service URL.",
          section: "verification",
          section_label: "Verification",
          value_type: "string",
          default: null,
          choices: [],
          depends_on: ["verification.critic_enabled"],
          prominence: "minor",
          secret: false,
          required: false,
        },
        {
          key: "verification.critic_model_name",
          label: "Critic Model Name",
          description: "Override the critic model name.",
          section: "verification",
          section_label: "Verification",
          value_type: "string",
          default: null,
          choices: [],
          depends_on: ["verification.critic_enabled"],
          prominence: "minor",
          secret: false,
          required: false,
        },
      ],
    },
    {
      key: "condenser",
      label: "Condenser",
      fields: [
        {
          description:
            "Enable the default LLM-based condenser to summarize long conversation histories.",
          key: "condenser.enable_default_condenser",
          label: "Enable default condenser",
          section: "condenser",
          section_label: "Condenser",
          value_type: "boolean",
          default: true,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: false,
          required: true,
        },
        {
          description:
            "Maximum number of tokens the condenser keeps after summarization. Leave blank for unlimited.",
          key: "condenser.condenser_max_size",
          label: "Condenser max size",
          section: "condenser",
          section_label: "Condenser",
          value_type: "integer",
          default: null,
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: false,
        },
      ],
    },
  ],
};

const MOCK_CONVERSATION_SETTINGS_SCHEMA: NonNullable<
  Settings["conversation_settings_schema"]
> = {
  model_name: "ConversationSettings",
  sections: [
    {
      key: "general",
      label: "General",
      fields: [
        {
          key: "max_iterations",
          label: "Max iterations",
          section: "general",
          description:
            "Maximum number of agent steps allowed before the conversation stops.",

          section_label: "General",
          value_type: "integer",
          default: 500,
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: true,
        },
      ],
    },
    {
      key: "verification",
      label: "Verification",
      fields: [
        {
          key: "confirmation_mode",
          label: "Confirmation mode",
          description:
            "Pause for confirmation before the agent performs high-risk actions.",

          section: "verification",
          section_label: "Verification",
          value_type: "boolean",
          default: false,
          choices: [],
          depends_on: [],
          prominence: "major",
          secret: false,
          required: true,
        },
        {
          key: "security_analyzer",
          label: "Security analyzer",
          description:
            "Choose how OpenHands should analyze actions before asking for confirmation.",

          section: "verification",
          section_label: "Verification",
          value_type: "string",
          default: "llm",
          choices: [
            { label: "llm", value: "llm" },
            { label: "none", value: "none" },
          ],
          depends_on: ["confirmation_mode"],
          prominence: "major",
          secret: false,
          required: false,
        },
      ],
    },
  ],
};

export const MOCK_DEFAULT_USER_SETTINGS: Settings = {
  ...DEFAULT_SETTINGS,
  provider_tokens_set: {},
  agent_settings_schema: MOCK_AGENT_SETTINGS_SCHEMA,
  agent_settings: {
    ...DEFAULT_AGENT_SETTINGS,
    verification: {
      critic_enabled: false,
      enable_iterative_refinement: false,
    },
    llm: {
      ...(llmDefaults ?? {}),
      api_key: null,
      model: DEFAULT_MODEL,
    },
    condenser: {
      enable_default_condenser: true,
      condenser_max_size: null,
    },
    enable_sub_agents: false,
    tool_concurrency_limit: 1,
  },
  conversation_settings_schema: MOCK_CONVERSATION_SETTINGS_SCHEMA,
  conversation_settings: {
    ...(DEFAULT_SETTINGS.conversation_settings ?? {}),
  },
};

const MOCK_USER_PREFERENCES: {
  settings: Settings | null;
} = {
  settings: structuredClone(MOCK_DEFAULT_USER_SETTINGS),
};

interface MockLlmProfile {
  name: string;
  config: Record<string, SettingsValue>;
  api_key_set: boolean;
}

const MOCK_LLM_PROFILES: {
  profiles: Map<string, MockLlmProfile>;
  activeProfile: string | null;
} = {
  profiles: new Map(),
  activeProfile: null,
};

let mockOpenAISubscriptionConnected = false;

const getProfileNameParam = (value: unknown): string =>
  decodeURIComponent(
    Array.isArray(value) ? String(value[0] ?? "") : String(value ?? ""),
  );

const profileToListItem = (profile: MockLlmProfile) => ({
  name: profile.name,
  model: typeof profile.config.model === "string" ? profile.config.model : null,
  base_url:
    typeof profile.config.base_url === "string"
      ? profile.config.base_url
      : null,
  api_key_set: profile.api_key_set,
});

const applyProfileToMockSettings = (profile: MockLlmProfile) => {
  const current =
    MOCK_USER_PREFERENCES.settings ||
    structuredClone(MOCK_DEFAULT_USER_SETTINGS);

  MOCK_USER_PREFERENCES.settings = {
    ...current,
    agent_settings: {
      ...(current.agent_settings ?? {}),
      llm: structuredClone(profile.config),
    },
    llm_api_key_set: profile.api_key_set,
  };
};

const buildProfileDetail = (
  profile: MockLlmProfile,
  exposeSecrets: string | null,
) => {
  const config = structuredClone(profile.config);

  if (profile.api_key_set && "api_key" in config) {
    if (exposeSecrets === "encrypted") {
      config.api_key = `gAAAAA_mock_encrypted_${profile.name}`;
    } else if (exposeSecrets === "plaintext") {
      // Keep the mock plaintext value.
    } else {
      config.api_key = null;
    }
  }

  return {
    name: profile.name,
    config,
    api_key_set: profile.api_key_set,
  };
};

const saveMockProfile = (name: string, request: SaveProfileRequest) => {
  const llm = structuredClone(
    (request.llm ?? {}) as Record<string, SettingsValue>,
  );

  if (typeof llm.model !== "string" || llm.model.trim().length === 0) {
    return null;
  }

  const profile: MockLlmProfile = {
    name,
    config: llm,
    api_key_set:
      typeof llm.api_key === "string" && llm.api_key.trim().length > 0,
  };

  MOCK_LLM_PROFILES.profiles.set(name, profile);

  if (MOCK_LLM_PROFILES.activeProfile === name) {
    applyProfileToMockSettings(profile);
  }

  return profile;
};

export const resetTestHandlersMockSettings = () => {
  MOCK_USER_PREFERENCES.settings = structuredClone(MOCK_DEFAULT_USER_SETTINGS);
  MOCK_LLM_PROFILES.profiles.clear();
  MOCK_LLM_PROFILES.activeProfile = null;
  mockOpenAISubscriptionConnected = false;
};

// Mock model data used by provider/model endpoints
const MOCK_MODELS = [
  "anthropic/claude-3.5",
  "anthropic/claude-sonnet-4-20250514",
  "anthropic/claude-sonnet-4-5-20250929",
  "anthropic/claude-haiku-4-5-20251001",
  "anthropic/claude-opus-4-5-20251101",
  "anthropic/claude-opus-4-8",
  "openai/gpt-3.5-turbo",
  "openai/gpt-5.5",
  "openai/gpt-4o",
  "openai/gpt-4o-mini",
  "openhands/claude-sonnet-4-20250514",
  "openhands/claude-sonnet-4-5-20250929",
  "openhands/claude-haiku-4-5-20251001",
  "openhands/claude-opus-4-5-20251101",
  "openhands/minimax-m2.7",
  "sambanova/Meta-Llama-3.1-8B-Instruct",
];

const MOCK_OPENAI_SUBSCRIPTION_MODELS = ["gpt-5.2", "gpt-5.3-codex"];

const MOCK_VERIFIED_MODELS = new Set([
  "anthropic/claude-opus-4-5-20251101",
  "anthropic/claude-opus-4-8",
  "anthropic/claude-sonnet-4-5-20250929",
  "openai/gpt-5.5",
  "openhands/claude-opus-4-5-20251101",
  "openhands/claude-sonnet-4-5-20250929",
  "openhands/minimax-m2.7",
]);

const MOCK_VERIFIED_PROVIDERS = [
  "openhands",
  "anthropic",
  "openai",
  "mistral",
  "gemini",
  "deepseek",
  "moonshot",
  "minimax",
];

const MOCK_MODEL_PROVIDERS = Array.from(
  new Set(
    MOCK_MODELS.map((model) => model.split("/")[0]).filter(
      (provider): provider is string => Boolean(provider),
    ),
  ),
);

const MOCK_VERIFIED_MODELS_BY_PROVIDER = MOCK_MODELS.reduce<
  Record<string, string[]>
>((acc, model) => {
  if (!MOCK_VERIFIED_MODELS.has(model)) return acc;

  const [provider, ...rest] = model.split("/");
  if (!provider || rest.length === 0) return acc;

  acc[provider] ??= [];
  acc[provider].push(rest.join("/"));
  return acc;
}, {});

const MOCK_AGENT_SERVER_VERSION = "1.29.3";

// --- Handlers for options/config/settings ---
// Uses wildcard "*" prefix to match both relative paths and absolute URLs
// (e.g., http://127.0.0.1:8000/api/...) since the code uses absolute URLs
// when VITE_BACKEND_BASE_URL is configured.

export const SETTINGS_HANDLERS = [
  http.get("*/server_info", async () =>
    HttpResponse.json({
      uptime: 0,
      idle_time: 0,
      version: MOCK_AGENT_SERVER_VERSION,
      usable_tools: [
        "terminal",
        "file_editor",
        "task_tracker",
        "browser_tool_set",
      ],
      agents: ["CodeActAgent"],
      default_agent: "CodeActAgent",
      models: MOCK_MODELS,
      security_analyzers: ["llm", "none"],
    }),
  ),

  http.get("*/api/llm/models", async () =>
    HttpResponse.json({ models: MOCK_MODELS }),
  ),

  http.get("*/api/llm/models/verified", async () =>
    HttpResponse.json({ models: MOCK_VERIFIED_MODELS_BY_PROVIDER }),
  ),

  http.get("*/api/llm/providers", async () =>
    HttpResponse.json({ providers: MOCK_MODEL_PROVIDERS }),
  ),

  http.get(`*${OPENAI_SUBSCRIPTION_MODELS_PATH}`, async () =>
    HttpResponse.json({
      vendor: OPENAI_SUBSCRIPTION_VENDOR,
      models: MOCK_OPENAI_SUBSCRIPTION_MODELS,
    }),
  ),

  http.get(`*${OPENAI_SUBSCRIPTION_STATUS_PATH}`, async () =>
    HttpResponse.json({
      connected: mockOpenAISubscriptionConnected,
      account_email: mockOpenAISubscriptionConnected
        ? "mock-chatgpt@example.com"
        : null,
      expires_at: null,
    }),
  ),

  http.post(`*${OPENAI_SUBSCRIPTION_DEVICE_START_PATH}`, async () =>
    HttpResponse.json({
      device_code: "mock-device-code",
      user_code: "MOCK-CODE",
      verification_uri: "https://auth.openai.com/activate",
      verification_uri_complete:
        "https://auth.openai.com/activate?user_code=MOCK-CODE",
      interval: 1,
      expires_in: 900,
    }),
  ),

  http.post(`*${OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH}`, async () => {
    mockOpenAISubscriptionConnected = true;
    return HttpResponse.json({
      connected: true,
      account_email: "mock-chatgpt@example.com",
      expires_at: null,
    });
  }),

  http.post(`*${OPENAI_SUBSCRIPTION_LOGOUT_PATH}`, async () => {
    mockOpenAISubscriptionConnected = false;
    return HttpResponse.json({ connected: false });
  }),

  // V0 (legacy) models endpoint – still used for default_model
  http.get("*/api/options/models", async () =>
    HttpResponse.json({
      models: MOCK_MODELS,
      verified_models: [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
      ],
      verified_providers: MOCK_VERIFIED_PROVIDERS,
      default_model: "openhands/minimax-m2.7",
    }),
  ),

  // V1 providers search
  http.get("*/api/v1/config/providers/search", async ({ request }) => {
    const url = new URL(request.url);
    const query = url.searchParams.get("query")?.toLowerCase();
    const verifiedEq = url.searchParams.get("verified__eq");

    // Build unique provider list from models
    const seen = new Set<string>();
    let providers: { name: string; verified: boolean }[] = [];
    for (const model of MOCK_MODELS) {
      const [providerName] = model.split("/");
      if (providerName && !seen.has(providerName)) {
        seen.add(providerName);
        providers.push({
          name: providerName,
          verified: MOCK_VERIFIED_PROVIDERS.includes(providerName),
        });
      }
    }

    if (query) {
      providers = providers.filter((p) => p.name.toLowerCase().includes(query));
    }
    if (verifiedEq !== null && verifiedEq !== undefined) {
      const wantVerified = verifiedEq === "true";
      providers = providers.filter((p) => p.verified === wantVerified);
    }

    return HttpResponse.json({ items: providers, next_page_id: null });
  }),

  // V1 models search
  http.get("*/api/v1/config/models/search", async ({ request }) => {
    const url = new URL(request.url);
    const query = url.searchParams.get("query")?.toLowerCase();
    const verifiedEq = url.searchParams.get("verified__eq");
    const providerEq = url.searchParams.get("provider__eq");

    let models = MOCK_MODELS.map((m) => {
      const [provider, ...rest] = m.split("/");
      const name = rest.join("/");
      return {
        provider: provider || null,
        name,
        verified: MOCK_VERIFIED_MODELS.has(m),
      };
    });

    if (providerEq) {
      models = models.filter((m) => m.provider === providerEq);
    }
    if (query) {
      models = models.filter((m) => m.name.toLowerCase().includes(query));
    }
    if (verifiedEq !== null && verifiedEq !== undefined) {
      const wantVerified = verifiedEq === "true";
      models = models.filter((m) => m.verified === wantVerified);
    }

    return HttpResponse.json({ items: models, next_page_id: null });
  }),

  http.get("*/api/options/security-analyzers", async () =>
    HttpResponse.json(["llm", "none"]),
  ),

  http.get("*/api/profiles", async () =>
    HttpResponse.json({
      profiles: Array.from(MOCK_LLM_PROFILES.profiles.values()).map(
        profileToListItem,
      ),
      active_profile: MOCK_LLM_PROFILES.activeProfile,
    }),
  ),

  http.get("*/api/profiles/:name", async ({ params, request }) => {
    const name = getProfileNameParam(params.name);
    const profile = MOCK_LLM_PROFILES.profiles.get(name);

    if (!profile) {
      return HttpResponse.json(
        { detail: `Profile '${name}' not found` },
        { status: 404 },
      );
    }

    return HttpResponse.json(
      buildProfileDetail(profile, request.headers.get("X-Expose-Secrets")),
    );
  }),

  http.post("*/api/profiles/:name", async ({ params, request }) => {
    const name = getProfileNameParam(params.name);
    const body = (await request.json()) as SaveProfileRequest | null;

    if (!body) {
      return HttpResponse.json({ detail: "Empty body" }, { status: 400 });
    }

    const profile = saveMockProfile(name, body);
    if (!profile) {
      return HttpResponse.json(
        { detail: "Profile requires llm.model" },
        { status: 400 },
      );
    }

    return HttpResponse.json(
      { name: profile.name, message: `Profile '${profile.name}' saved` },
      { status: 201 },
    );
  }),

  http.delete("*/api/profiles/:name", async ({ params }) => {
    const name = getProfileNameParam(params.name);
    MOCK_LLM_PROFILES.profiles.delete(name);

    if (MOCK_LLM_PROFILES.activeProfile === name) {
      MOCK_LLM_PROFILES.activeProfile = null;
    }

    return HttpResponse.json({
      name,
      message: `Profile '${name}' deleted`,
    });
  }),

  http.post("*/api/profiles/:name/rename", async ({ params, request }) => {
    const name = getProfileNameParam(params.name);
    const body = (await request.json()) as { new_name?: string } | null;
    const newName = body?.new_name?.trim() ?? "";
    const profile = MOCK_LLM_PROFILES.profiles.get(name);

    if (!profile) {
      return HttpResponse.json(
        { detail: `Profile '${name}' not found` },
        { status: 404 },
      );
    }

    if (!newName) {
      return HttpResponse.json(
        { detail: "new_name is required" },
        { status: 400 },
      );
    }

    if (newName !== name && MOCK_LLM_PROFILES.profiles.has(newName)) {
      return HttpResponse.json(
        { detail: `Profile '${newName}' already exists` },
        { status: 409 },
      );
    }

    MOCK_LLM_PROFILES.profiles.delete(name);
    const renamedProfile = { ...profile, name: newName };
    MOCK_LLM_PROFILES.profiles.set(newName, renamedProfile);

    if (MOCK_LLM_PROFILES.activeProfile === name) {
      MOCK_LLM_PROFILES.activeProfile = newName;
      applyProfileToMockSettings(renamedProfile);
    }

    return HttpResponse.json({
      name: newName,
      message: `Profile '${name}' renamed to '${newName}'`,
    });
  }),

  http.post("*/api/profiles/:name/activate", async ({ params }) => {
    const name = getProfileNameParam(params.name);
    const profile = MOCK_LLM_PROFILES.profiles.get(name);

    if (!profile) {
      return HttpResponse.json(
        { detail: `Profile '${name}' not found` },
        { status: 404 },
      );
    }

    MOCK_LLM_PROFILES.activeProfile = name;
    applyProfileToMockSettings(profile);

    return HttpResponse.json({
      name,
      message: `Profile '${name}' activated and applied to current settings`,
      llm_applied: true,
    });
  }),

  http.get("*/api/v1/web-client/config", () => {
    const config: WebClientConfig = {
      feature_flags: {
        hide_llm_settings: false,
        hide_users_page: false,
      },
      providers_configured: [],
      maintenance_start_time: null,
      recaptcha_site_key: null,
      faulty_models: [],
      error_message: null,
      updated_at: new Date().toISOString(),
    };

    return HttpResponse.json(config);
  }),

  http.get("*/api/settings/conversation-schema", async () => {
    await delay();
    return HttpResponse.json(MOCK_CONVERSATION_SETTINGS_SCHEMA);
  }),

  http.get("*/api/v1/settings/conversation-schema", async () => {
    await delay();
    return HttpResponse.json(MOCK_CONVERSATION_SETTINGS_SCHEMA);
  }),

  http.get("*/api/v1/settings", async () => {
    await delay();
    const { settings } = MOCK_USER_PREFERENCES;

    if (!settings) return HttpResponse.json(null, { status: 404 });

    return HttpResponse.json(settings);
  }),

  // New settings API endpoints (GET /api/settings with X-Expose-Secrets header support)
  http.get("*/api/settings", async ({ request }) => {
    // Exclude sub-paths like /api/settings/agent-schema (handled by separate handlers)
    const url = new URL(request.url);
    const pathParts = url.pathname.split("/").filter(Boolean);
    if (pathParts.length > 2) return undefined;

    await delay();
    const { settings } = MOCK_USER_PREFERENCES;

    const DEFAULT_APP_PREFERENCES = {
      language: null,
      user_consents_to_analytics: null,
      enable_sound_notifications: null,
      git_user_name: null,
      git_user_email: null,
      title_llm_profile: null,
      disabled_skills: [],
    };

    if (!settings) {
      return HttpResponse.json({
        agent_settings: {},
        conversation_settings: {},
        llm_api_key_is_set: false,
        misc_settings: { app_preferences: DEFAULT_APP_PREFERENCES },
      });
    }

    const exposeSecrets = request.headers.get("X-Expose-Secrets");

    // Build agent_settings, handling secrets based on header
    const agentSettings = structuredClone(
      settings.agent_settings ?? {},
    ) as Record<string, unknown>;
    const llm = agentSettings.llm as Record<string, unknown> | undefined;
    if (llm?.api_key) {
      if (exposeSecrets === "encrypted") {
        // Return a mock "encrypted" value
        llm.api_key = `gAAAAA_mock_encrypted_${String(llm.api_key).slice(0, 8)}`;
      } else if (exposeSecrets === "plaintext") {
        // Keep as-is (plaintext)
      } else {
        // Redact
        llm.api_key = "**********";
      }
    }

    const llmApiKeySet =
      !!settings.llm_api_key_set ||
      (!!(settings.agent_settings as Record<string, unknown> | undefined)
        ?.llm &&
        !!(
          (settings.agent_settings as Record<string, unknown>).llm as Record<
            string,
            unknown
          >
        )?.api_key);

    // Reuse the persisted misc_settings.app_preferences for repeat fetches,
    // but always fall back to the default-empty block so the GUI sees a
    // deterministic shape on first read.
    const storedMisc = (settings as Record<string, unknown>).misc_settings as
      | { app_preferences?: Record<string, unknown> }
      | undefined;
    const appPreferences = {
      ...DEFAULT_APP_PREFERENCES,
      ...(storedMisc?.app_preferences ?? {}),
    };

    return HttpResponse.json({
      agent_settings: agentSettings,
      conversation_settings: settings.conversation_settings ?? {},
      llm_api_key_is_set: llmApiKeySet,
      misc_settings: { app_preferences: appPreferences },
    });
  }),

  // PATCH /api/settings - incremental updates
  http.patch("*/api/settings", async ({ request }) => {
    await delay();
    const body = (await request.json()) as {
      agent_settings_diff?: Record<string, unknown>;
      conversation_settings_diff?: Record<string, SettingsValue>;
      misc_settings_diff?: {
        app_preferences?: Record<string, unknown>;
      };
    } | null;

    if (!body) {
      return HttpResponse.json({ error: "Empty body" }, { status: 400 });
    }

    if (
      !body.agent_settings_diff &&
      !body.conversation_settings_diff &&
      !body.misc_settings_diff
    ) {
      return HttpResponse.json(
        {
          error:
            "At least one of agent_settings_diff, conversation_settings_diff, or misc_settings_diff must be provided",
        },
        { status: 400 },
      );
    }

    const current =
      MOCK_USER_PREFERENCES.settings ||
      structuredClone(MOCK_DEFAULT_USER_SETTINGS);
    const nextSettings: Settings = { ...current };

    if (body.agent_settings_diff) {
      const merged = deepMerge(
        (current.agent_settings ?? {}) as Record<string, unknown>,
        body.agent_settings_diff,
      );
      nextSettings.agent_settings = merged as Settings["agent_settings"];

      // Sync llm_api_key_set
      const llm = merged.llm as Record<string, unknown> | undefined;
      if (
        llm?.api_key &&
        typeof llm.api_key === "string" &&
        llm.api_key.trim().length > 0
      ) {
        nextSettings.llm_api_key_set = true;
      }
    }

    if (body.conversation_settings_diff) {
      nextSettings.conversation_settings = {
        ...(current.conversation_settings ?? {}),
        ...body.conversation_settings_diff,
      };
    }

    if (body.misc_settings_diff) {
      const existingMisc = (current as Record<string, unknown>)
        .misc_settings as
        | { app_preferences?: Record<string, unknown> }
        | undefined;
      // Deep-merge: nested `app_preferences` overlays field-by-field;
      // `disabled_skills` lists are replaced wholesale. This mirrors the
      // SDK's `_deep_merge` behaviour for the two-level shape currently
      // stored in `misc_settings`.
      const nextMisc: { app_preferences?: Record<string, unknown> } = {
        ...(existingMisc ?? {}),
      };
      if (body.misc_settings_diff.app_preferences) {
        nextMisc.app_preferences = {
          ...(existingMisc?.app_preferences ?? {}),
          ...body.misc_settings_diff.app_preferences,
        };
      }
      (nextSettings as Record<string, unknown>).misc_settings = nextMisc;
    }

    MOCK_USER_PREFERENCES.settings = nextSettings;

    // Return the updated settings (without secrets exposed)
    return HttpResponse.json({
      agent_settings: nextSettings.agent_settings ?? {},
      conversation_settings: nextSettings.conversation_settings ?? {},
      llm_api_key_is_set: nextSettings.llm_api_key_set ?? false,
      misc_settings: ((nextSettings as Record<string, unknown>)
        .misc_settings as
        | { app_preferences?: Record<string, unknown> }
        | undefined) ?? { app_preferences: {} },
    });
  }),

  http.post("*/api/mcp/test", async () => {
    await delay();
    return HttpResponse.json({ ok: true, tools: ["mock_tool"] });
  }),

  http.get("*/api/settings/agent-schema", async () => {
    await delay();
    return HttpResponse.json(MOCK_AGENT_SETTINGS_SCHEMA);
  }),

  http.get("*/api/v1/settings/agent-schema", async () => {
    await delay();
    return HttpResponse.json(MOCK_AGENT_SETTINGS_SCHEMA);
  }),

  http.post("*/api/v1/settings", async ({ request }) => {
    await delay();
    const body = (await request.json()) as Record<string, unknown> | null;

    if (body) {
      const current =
        MOCK_USER_PREFERENCES.settings ||
        structuredClone(MOCK_DEFAULT_USER_SETTINGS);

      if ("agent_settings" in body || "conversation_settings" in body) {
        return HttpResponse.json(
          {
            error: "Use *_diff nested settings payloads",
            keys: ["agent_settings", "conversation_settings"].filter(
              (key) => key in body,
            ),
          },
          { status: 422 },
        );
      }

      const nextSettings: Settings = { ...current };

      const agentSettingsPatch = body.agent_settings_diff as
        | Record<string, unknown>
        | undefined;
      if (agentSettingsPatch) {
        const merged = deepMerge(
          (current.agent_settings ?? {}) as Record<string, unknown>,
          agentSettingsPatch,
        );
        nextSettings.agent_settings = merged as Settings["agent_settings"];
      }

      const conversationSettingsPatch = body.conversation_settings_diff as
        | Record<string, SettingsValue>
        | undefined;
      if (conversationSettingsPatch) {
        nextSettings.conversation_settings = {
          ...(current.conversation_settings ?? {}),
          ...conversationSettingsPatch,
        };
      }

      // Apply top-level fields (excluding nested settings)
      for (const [key, value] of Object.entries(body)) {
        if (
          key !== "agent_settings_diff" &&
          key !== "conversation_settings_diff" &&
          key !== "agent_settings_schema" &&
          key !== "conversation_settings_schema"
        ) {
          (nextSettings as Record<string, unknown>)[key] = value;
        }
      }

      MOCK_USER_PREFERENCES.settings = nextSettings;
      return HttpResponse.json(null, { status: 200 });
    }

    return HttpResponse.json(null, { status: 400 });
  }),

  // POST /api/mcp/test – MCP server connectivity check before install.
  // Returns ok:true so the install modal can proceed to save and close.
  http.post("*/api/mcp/test", async () => {
    await delay();
    return HttpResponse.json({ ok: true, tools: [] });
  }),
];
