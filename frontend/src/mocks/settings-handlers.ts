import { http, delay, HttpResponse } from "msw";
import { WebClientConfig } from "#/api/option-service/option.types";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Provider, Settings, SettingsValue } from "#/types/settings";

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
  app_mode: "oss",
  posthog_client_key: "test-posthog-key",
  feature_flags: {
    enable_billing: false,
    hide_llm_settings: false,
    enable_jira: false,
    enable_jira_dc: false,
    enable_linear: false,
    hide_users_page: false,
    hide_billing_page: false,
    hide_integrations_page: false,
    ...overrides.feature_flags,
  },
  providers_configured: [],
  maintenance_start_time: null,
  auth_url: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: new Date().toISOString(),
  github_app_slug: null,
  ...overrides,
});

const MOCK_AGENT_SETTINGS_SCHEMA: NonNullable<
  Settings["agent_settings_schema"]
> = {
  model_name: "AgentSettings",
  sections: [
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
      ],
    },
    {
      key: "critic",
      label: "Critic",
      fields: [
        {
          description:
            "Enable an additional critic pass to review the agent's work.",

          key: "critic.enabled",
          label: "Enable critic",
          section: "critic",
          section_label: "Critic",
          value_type: "boolean",
          default: false,
          choices: [],
          depends_on: [],
          prominence: "critical",
          secret: false,
          required: true,
        },
        {
          description: "Choose when the critic should review and intervene.",

          key: "critic.mode",
          label: "Mode",
          section: "critic",
          section_label: "Critic",
          value_type: "string",
          default: "finish_and_message",
          choices: [
            { label: "finish_and_message", value: "finish_and_message" },
            { label: "all_actions", value: "all_actions" },
          ],
          depends_on: ["critic.enabled"],
          prominence: "minor",
          secret: false,
          required: true,
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
    critic: {
      mode: "finish_and_message",
      enabled: false,
    },
    llm: {
      ...(llmDefaults ?? {}),
      api_key: null,
      model: DEFAULT_MODEL,
    },
  },
  conversation_settings_schema: MOCK_CONVERSATION_SETTINGS_SCHEMA,
  conversation_settings: {
    ...(DEFAULT_SETTINGS.conversation_settings ?? {}),
  },
};

const MOCK_USER_PREFERENCES: {
  settings: Settings | null;
} = {
  settings: null,
};

export const resetTestHandlersMockSettings = () => {
  MOCK_USER_PREFERENCES.settings = structuredClone(MOCK_DEFAULT_USER_SETTINGS);
};

// Mock model data used by both V0 and V1 endpoints
const MOCK_MODELS = [
  "anthropic/claude-3.5",
  "anthropic/claude-sonnet-4-20250514",
  "anthropic/claude-sonnet-4-5-20250929",
  "anthropic/claude-haiku-4-5-20251001",
  "anthropic/claude-opus-4-5-20251101",
  "openai/gpt-3.5-turbo",
  "openai/gpt-4o",
  "openai/gpt-4o-mini",
  "openhands/claude-sonnet-4-20250514",
  "openhands/claude-sonnet-4-5-20250929",
  "openhands/claude-haiku-4-5-20251001",
  "openhands/claude-opus-4-5-20251101",
  "openhands/minimax-m2.7",
  "sambanova/Meta-Llama-3.1-8B-Instruct",
];

const MOCK_VERIFIED_MODELS = new Set([
  "anthropic/claude-opus-4-5-20251101",
  "anthropic/claude-sonnet-4-5-20250929",
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

// --- Handlers for options/config/settings ---

export const SETTINGS_HANDLERS = [
  // V0 (legacy) models endpoint – still used for default_model
  http.get("/api/options/models", async () =>
    HttpResponse.json({
      models: MOCK_MODELS,
      verified_models: [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
      ],
      verified_providers: MOCK_VERIFIED_PROVIDERS,
      default_model: "openhands/claude-opus-4-5-20251101",
    }),
  ),

  // V1 providers search
  http.get("/api/v1/config/providers/search", async ({ request }) => {
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
  http.get("/api/v1/config/models/search", async ({ request }) => {
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

  http.get("/api/options/security-analyzers", async () =>
    HttpResponse.json(["llm", "none"]),
  ),

  http.get("/api/v1/web-client/config", () => {
    const mockSaas = import.meta.env.VITE_MOCK_SAAS === "true";

    const config: WebClientConfig = {
      app_mode: mockSaas ? "saas" : "oss",
      posthog_client_key: "fake-posthog-client-key",
      feature_flags: {
        enable_billing: mockSaas,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
        hide_users_page: false,
        hide_billing_page: false,
        hide_integrations_page: false,
      },
      providers_configured: [],
      maintenance_start_time: null,
      auth_url: null,
      recaptcha_site_key: null,
      faulty_models: [],
      error_message: null,
      updated_at: new Date().toISOString(),
      github_app_slug: mockSaas ? "openhands" : null,
    };

    return HttpResponse.json(config);
  }),

  http.get("/api/v1/settings/conversation-schema", async () => {
    await delay();
    return HttpResponse.json(MOCK_CONVERSATION_SETTINGS_SCHEMA);
  }),

  http.get("/api/v1/settings", async () => {
    await delay();
    const { settings } = MOCK_USER_PREFERENCES;

    if (!settings) return HttpResponse.json(null, { status: 404 });

    return HttpResponse.json(settings);
  }),

  http.get("/api/v1/settings/agent-schema", async () => {
    await delay();
    return HttpResponse.json(MOCK_AGENT_SETTINGS_SCHEMA);
  }),

  http.post("/api/v1/settings", async ({ request }) => {
    await delay();
    const body = (await request.json()) as Record<string, unknown> | null;

    if (body) {
      const current =
        MOCK_USER_PREFERENCES.settings ||
        structuredClone(MOCK_DEFAULT_USER_SETTINGS);

      const nextSettings: Settings = { ...current };

      // Deep-merge nested agent_settings
      if (body.agent_settings && typeof body.agent_settings === "object") {
        const merged = deepMerge(
          (current.agent_settings ?? {}) as Record<string, unknown>,
          body.agent_settings as Record<string, unknown>,
        );
        nextSettings.agent_settings = merged as Settings["agent_settings"];
      }

      // Deep-merge nested conversation_settings
      if (
        body.conversation_settings &&
        typeof body.conversation_settings === "object"
      ) {
        nextSettings.conversation_settings = {
          ...(current.conversation_settings ?? {}),
          ...(body.conversation_settings as Record<string, SettingsValue>),
        };
      }

      // Apply top-level fields (excluding nested settings)
      for (const [key, value] of Object.entries(body)) {
        if (
          key !== "agent_settings" &&
          key !== "conversation_settings" &&
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

  http.post("/api/add-git-providers", async ({ request }) => {
    const body = await request.json();

    if (typeof body === "object" && body?.provider_tokens) {
      const rawTokens = body.provider_tokens as Record<
        string,
        { token?: string }
      >;

      const providerTokensSet: Partial<Record<Provider, string | null>> =
        Object.fromEntries(
          Object.entries(rawTokens)
            .filter(([, val]) => val?.token)
            .map(([provider]) => [provider as Provider, ""]),
        );

      MOCK_USER_PREFERENCES.settings = {
        ...(MOCK_USER_PREFERENCES.settings || MOCK_DEFAULT_USER_SETTINGS),
        provider_tokens_set: providerTokensSet,
      };

      return HttpResponse.json(true, { status: 200 });
    }

    return HttpResponse.json(null, { status: 400 });
  }),
];
