import { http, delay, HttpResponse } from "msw";
import { GetConfigResponse } from "#/api/option-service/option.types";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Provider, Settings } from "#/types/settings";

export const MOCK_DEFAULT_USER_SETTINGS: Settings = {
  llm_model: DEFAULT_SETTINGS.llm_model,
  llm_base_url: DEFAULT_SETTINGS.llm_base_url,
  llm_api_key: null,
  llm_api_key_set: DEFAULT_SETTINGS.llm_api_key_set,
  search_api_key_set: DEFAULT_SETTINGS.search_api_key_set,
  agent: DEFAULT_SETTINGS.agent,
  language: DEFAULT_SETTINGS.language,
  confirmation_mode: DEFAULT_SETTINGS.confirmation_mode,
  security_analyzer: DEFAULT_SETTINGS.security_analyzer,
  remote_runtime_resource_factor:
    DEFAULT_SETTINGS.remote_runtime_resource_factor,
  provider_tokens_set: {},
  enable_default_condenser: DEFAULT_SETTINGS.enable_default_condenser,
  condenser_max_size: DEFAULT_SETTINGS.condenser_max_size,
  enable_sound_notifications: DEFAULT_SETTINGS.enable_sound_notifications,
  enable_proactive_conversation_starters:
    DEFAULT_SETTINGS.enable_proactive_conversation_starters,
  enable_solvability_analysis: DEFAULT_SETTINGS.enable_solvability_analysis,
  user_consents_to_analytics: DEFAULT_SETTINGS.user_consents_to_analytics,
  max_budget_per_task: DEFAULT_SETTINGS.max_budget_per_task,
};

const MOCK_USER_PREFERENCES: {
  settings: Settings | null;
} = {
  settings: null,
};

// Reset mock
export const resetTestHandlersMockSettings = () => {
  MOCK_USER_PREFERENCES.settings = MOCK_DEFAULT_USER_SETTINGS;
};

// --- Handlers for options/config/settings ---

export const SETTINGS_HANDLERS = [
  http.get("/api/options/models", async () =>
    HttpResponse.json([
      "gpt-3.5-turbo",
      "gpt-4o",
      "gpt-4o-mini",
      "anthropic/claude-3.5",
      "anthropic/claude-sonnet-4-20250514",
      "anthropic/claude-sonnet-4-5-20250929",
      "anthropic/claude-haiku-4-5-20251001",
      "openhands/claude-sonnet-4-20250514",
      "openhands/claude-sonnet-4-5-20250929",
      "openhands/claude-haiku-4-5-20251001",
      "sambanova/Meta-Llama-3.1-8B-Instruct",
    ]),
  ),

  http.get("/api/options/agents", async () =>
    HttpResponse.json(["CodeActAgent", "CoActAgent"]),
  ),

  http.get("/api/options/security-analyzers", async () =>
    HttpResponse.json(["llm", "none"]),
  ),

  http.get("/api/options/config", () => {
    const mockSaas = import.meta.env.VITE_MOCK_SAAS === "true";

    const config: GetConfigResponse = {
      APP_MODE: mockSaas ? "saas" : "oss",
      GITHUB_CLIENT_ID: "fake-github-client-id",
      POSTHOG_CLIENT_KEY: "fake-posthog-client-key",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
        HIDE_LLM_SETTINGS: mockSaas,
        ENABLE_JIRA: false,
        ENABLE_JIRA_DC: false,
        ENABLE_LINEAR: false,
      },
      // Uncomment the following to test the maintenance banner
      // MAINTENANCE: {
      //   startTime: "2024-01-15T10:00:00-05:00", // EST timestamp
      // },
    };

    return HttpResponse.json(config);
  }),

  http.get("/api/settings", async () => {
    await delay();
    const { settings } = MOCK_USER_PREFERENCES;

    if (!settings) return HttpResponse.json(null, { status: 404 });

    return HttpResponse.json(settings);
  }),

  http.post("/api/settings", async ({ request }) => {
    await delay();
    const body = await request.json();

    if (body) {
      const current = MOCK_USER_PREFERENCES.settings || {
        ...MOCK_DEFAULT_USER_SETTINGS,
      };

      MOCK_USER_PREFERENCES.settings = {
        ...current,
        ...(body as Partial<Settings>),
      };

      return HttpResponse.json(null, { status: 200 });
    }

    return HttpResponse.json(null, { status: 400 });
  }),

  http.post("/api/reset-settings", async () => {
    await delay();
    MOCK_USER_PREFERENCES.settings = { ...MOCK_DEFAULT_USER_SETTINGS };
    return HttpResponse.json(null, { status: 200 });
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
            .filter(([, val]) => val && val.token)
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
