import { http, delay, HttpResponse } from "msw";
import { WebClientConfig } from "#/api/option-service/option.types";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Provider, Settings } from "#/types/settings";

/**
 * Creates a mock WebClientConfig with all required fields.
 * Use this helper to create test config objects with sensible defaults.
 */
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
      // Uncomment the following to test the maintenance banner
      // maintenance_start_time: "2024-01-15T10:00:00-05:00", // EST timestamp
      auth_url: null,
      recaptcha_site_key: null,
      faulty_models: [],
      error_message: null,
      updated_at: new Date().toISOString(),
      github_app_slug: mockSaas ? "openhands" : null,
    };

    return HttpResponse.json(config);
  }),

  http.get("/api/v1/settings", async () => {
    await delay();
    const { settings } = MOCK_USER_PREFERENCES;

    if (!settings) return HttpResponse.json(null, { status: 404 });

    return HttpResponse.json(settings);
  }),

  http.post("/api/v1/settings", async ({ request }) => {
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
