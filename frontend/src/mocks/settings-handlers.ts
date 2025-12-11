import { http, delay, HttpResponse } from "msw";
import {
  ApiSettings,
  PostApiSettings,
} from "#/api/settings-service/settings.types";
import { GetConfigResponse } from "#/api/option-service/option.types";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Provider } from "#/types/settings";

export const MOCK_DEFAULT_USER_SETTINGS: ApiSettings | PostApiSettings = {
  llm_model: DEFAULT_SETTINGS.LLM_MODEL,
  llm_base_url: DEFAULT_SETTINGS.LLM_BASE_URL,
  llm_api_key: null,
  llm_api_key_set: DEFAULT_SETTINGS.LLM_API_KEY_SET,
  search_api_key_set: DEFAULT_SETTINGS.SEARCH_API_KEY_SET,
  agent: DEFAULT_SETTINGS.AGENT,
  language: DEFAULT_SETTINGS.LANGUAGE,
  confirmation_mode: DEFAULT_SETTINGS.CONFIRMATION_MODE,
  security_analyzer: DEFAULT_SETTINGS.SECURITY_ANALYZER,
  remote_runtime_resource_factor:
    DEFAULT_SETTINGS.REMOTE_RUNTIME_RESOURCE_FACTOR,
  provider_tokens_set: {},
  enable_default_condenser: DEFAULT_SETTINGS.ENABLE_DEFAULT_CONDENSER,
  condenser_max_size: DEFAULT_SETTINGS.CONDENSER_MAX_SIZE,
  enable_sound_notifications: DEFAULT_SETTINGS.ENABLE_SOUND_NOTIFICATIONS,
  enable_proactive_conversation_starters:
    DEFAULT_SETTINGS.ENABLE_PROACTIVE_CONVERSATION_STARTERS,
  enable_solvability_analysis: DEFAULT_SETTINGS.ENABLE_SOLVABILITY_ANALYSIS,
  user_consents_to_analytics: DEFAULT_SETTINGS.USER_CONSENTS_TO_ANALYTICS,
  max_budget_per_task: DEFAULT_SETTINGS.MAX_BUDGET_PER_TASK,
};

const MOCK_USER_PREFERENCES: {
  settings: ApiSettings | PostApiSettings | null;
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
        ...(body as Partial<ApiSettings>),
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
