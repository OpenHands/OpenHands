import { delay, http, HttpResponse } from "msw";
import { GetConfigResponse } from "#/api/option-service/option.types";
import { Conversation, ResultSet } from "#/api/open-hands.types";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { STRIPE_BILLING_HANDLERS } from "./billing-handlers";
import { Provider } from "#/types/settings";
import {
  ApiSettings,
  PostApiSettings,
} from "#/settings-service/settings.types";
import { FILE_SERVICE_HANDLERS } from "./file-service-handlers";
import { GitUser } from "#/types/git";
import { TASK_SUGGESTIONS_HANDLERS } from "./task-suggestions-handlers";
import { SECRETS_HANDLERS } from "./secrets-handlers";
import { ORG_HANDLERS } from "./org-handlers";
import { GIT_REPOSITORY_HANDLERS } from "./git-repository-handlers";

const IS_MOCK_SAAS = import.meta.env.VITE_MOCK_SAAS === "true";

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
  settings: IS_MOCK_SAAS
    ? {
        ...MOCK_DEFAULT_USER_SETTINGS,
        provider_tokens_set: { github: "asd", gitlab: "def" },
      }
    : null,
};

/**
 * Set the user settings to the default settings
 *
 * Useful for resetting the settings in tests
 */
export const resetTestHandlersMockSettings = () => {
  MOCK_USER_PREFERENCES.settings = MOCK_DEFAULT_USER_SETTINGS;
};

const conversations: Conversation[] = [
  {
    conversation_id: "1",
    title: "My New Project",
    selected_repository: null,
    git_provider: null,
    selected_branch: null,
    last_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    status: "RUNNING",
    runtime_status: "STATUS$READY",
    url: null,
    session_api_key: null,
  },
  {
    conversation_id: "2",
    title: "Repo Testing",
    selected_repository: "octocat/hello-world",
    git_provider: "github",
    selected_branch: null,
    // 2 days ago
    last_updated_at: new Date(
      Date.now() - 2 * 24 * 60 * 60 * 1000,
    ).toISOString(),
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    status: "STOPPED",
    runtime_status: null,
    url: null,
    session_api_key: null,
  },
  {
    conversation_id: "3",
    title: "Another Project",
    selected_repository: "octocat/earth",
    git_provider: null,
    selected_branch: "main",
    // 5 days ago
    last_updated_at: new Date(
      Date.now() - 5 * 24 * 60 * 60 * 1000,
    ).toISOString(),
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    status: "STOPPED",
    runtime_status: null,
    url: null,
    session_api_key: null,
  },
];

const CONVERSATIONS = new Map<string, Conversation>(
  conversations.map((conversation) => [
    conversation.conversation_id,
    conversation,
  ]),
);

const openHandsHandlers = [
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

  http.post("http://localhost:3001/api/submit-feedback", async () => {
    await delay(1200);

    return HttpResponse.json({
      statusCode: 200,
      body: { message: "Success", link: "fake-url.com", password: "abc123" },
    });
  }),
];

export const handlers = [
  ...ORG_HANDLERS,
  ...STRIPE_BILLING_HANDLERS,
  ...FILE_SERVICE_HANDLERS,
  ...TASK_SUGGESTIONS_HANDLERS,
  ...SECRETS_HANDLERS,
  ...GIT_REPOSITORY_HANDLERS,
  ...openHandsHandlers,
  http.get("/api/user/info", () => {
    const user: GitUser = {
      id: "1",
      login: "octocat",
      avatar_url: "https://avatars.githubusercontent.com/u/583231?v=4",
      company: "GitHub",
      email: "placeholder@placeholder.placeholder",
      name: "monalisa octocat",
    };

    return HttpResponse.json(user);
  }),
  http.post("http://localhost:3001/api/submit-feedback", async () =>
    HttpResponse.json({ statusCode: 200 }, { status: 200 }),
  ),
  http.post("https://us.i.posthog.com/e", async () =>
    HttpResponse.json(null, { status: 200 }),
  ),
  http.get("/api/options/config", () => {
    const config: GetConfigResponse = {
      APP_MODE: IS_MOCK_SAAS ? "saas" : "oss",
      GITHUB_CLIENT_ID: "fake-github-client-id",
      POSTHOG_CLIENT_KEY: "fake-posthog-client-key",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
        HIDE_LLM_SETTINGS: IS_MOCK_SAAS,
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
      // Persist new values over current/mock defaults
      MOCK_USER_PREFERENCES.settings = {
        ...current,
        ...(body as Partial<ApiSettings>),
      };
      return HttpResponse.json(null, { status: 200 });
    }

    return HttpResponse.json(null, { status: 400 });
  }),

  http.post("/api/authenticate", async () =>
    HttpResponse.json({ message: "Authenticated" }),
  ),

  http.get("/api/conversations", async () => {
    const values = Array.from(CONVERSATIONS.values());
    const results: ResultSet<Conversation> = {
      results: values,
      next_page_id: null,
    };

    return HttpResponse.json(results, { status: 200 });
  }),

  http.delete("/api/conversations/:conversationId", async ({ params }) => {
    const { conversationId } = params;

    if (typeof conversationId === "string") {
      CONVERSATIONS.delete(conversationId);
      return HttpResponse.json(null, { status: 200 });
    }

    return HttpResponse.json(null, { status: 404 });
  }),

  http.patch(
    "/api/conversations/:conversationId",
    async ({ params, request }) => {
      const { conversationId } = params;

      if (typeof conversationId === "string") {
        const conversation = CONVERSATIONS.get(conversationId);

        if (conversation) {
          const body = await request.json();
          if (typeof body === "object" && body?.title) {
            CONVERSATIONS.set(conversationId, {
              ...conversation,
              title: body.title,
            });
            return HttpResponse.json(null, { status: 200 });
          }
        }
      }

      return HttpResponse.json(null, { status: 404 });
    },
  ),

  http.post("/api/conversations", async () => {
    await delay();

    const conversation: Conversation = {
      conversation_id: (Math.random() * 100).toString(),
      title: "New Conversation",
      selected_repository: null,
      git_provider: null,
      selected_branch: null,
      last_updated_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
      status: "RUNNING",
      runtime_status: "STATUS$READY",
      url: null,
      session_api_key: null,
    };

    CONVERSATIONS.set(conversation.conversation_id, conversation);
    return HttpResponse.json(conversation, { status: 201 });
  }),

  http.get("/api/conversations/:conversationId", async ({ params }) => {
    const { conversationId } = params;

    if (typeof conversationId === "string") {
      const project = CONVERSATIONS.get(conversationId);

      if (project) {
        return HttpResponse.json(project, { status: 200 });
      }
    }

    return HttpResponse.json(null, { status: 404 });
  }),

  http.post("/api/logout", () => HttpResponse.json(null, { status: 200 })),

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

      const newSettings = {
        ...(MOCK_USER_PREFERENCES.settings ?? MOCK_DEFAULT_USER_SETTINGS),
        provider_tokens_set: providerTokensSet,
      };
      MOCK_USER_PREFERENCES.settings = newSettings;

      return HttpResponse.json(true, { status: 200 });
    }

    return HttpResponse.json(null, { status: 400 });
  }),
];
