import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  fetchCloudConversationSettingsSchema,
  fetchCloudSettings,
  fetchCloudSettingsSchema,
  saveCloudSettings,
} from "#/api/cloud/settings-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { server } from "#/mocks/node";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:3000",
  apiKey: "local-key",
  kind: "local",
};

const SETTINGS_URL = `${cloudBackend.host}/api/v1/settings`;

// The cloud client talks over `fetch` (intercepted by MSW), so we capture the
// intercepted requests to assert on route, method, auth header, and — for
// writes — the exact JSON body that reached the wire.
let capturedRequests: Request[] = [];
let capturedBodies: Array<Record<string, unknown>> = [];

/** Stub `GET <path>` on the cloud backend, recording the request. */
function stubGet(path: string, data: unknown, status = 200) {
  server.use(
    http.get(`${cloudBackend.host}${path}`, ({ request }) => {
      capturedRequests.push(request);
      return HttpResponse.json(data, { status });
    }),
  );
}

/** Stub `POST <path>` on the cloud backend, recording request + JSON body. */
function stubPost(path: string, data: unknown = {}, status = 200) {
  server.use(
    http.post(`${cloudBackend.host}${path}`, async ({ request }) => {
      capturedRequests.push(request);
      capturedBodies.push(
        (await request.clone().json()) as Record<string, unknown>,
      );
      return HttpResponse.json(data, { status });
    }),
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  capturedRequests = [];
  capturedBodies = [];
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("cloud settings", () => {
  it("fetchCloudSettings preserves provider_tokens_set so the repo chain can fire", async () => {
    stubGet("/api/v1/settings", {
      llm_model: "anthropic/claude-3-5-sonnet",
      llm_base_url: "https://api.anthropic.com",
      llm_api_key_set: true,
      agent: "CodeActAgent",
      confirmation_mode: true,
      security_analyzer: "llm",
      max_iterations: 30,
      provider_tokens_set: { github: "***" },
    });

    const result = await fetchCloudSettings();

    const request = capturedRequests[0]!;
    expect(request.method).toBe("GET");
    expect(request.url).toBe(SETTINGS_URL);
    expect(request.headers.get("authorization")).toBe("Bearer bearer-token");

    // provider_tokens_set must round-trip — it's what drives
    // useUserProviders → useAppInstallations → useGitRepositories.
    expect(result.provider_tokens_set).toEqual({ github: "***" });

    // Top-level cloud fields are preserved as-is.
    expect(result.llm_model).toBe("anthropic/claude-3-5-sonnet");
    expect(result.llm_api_key_set).toBe(true);
    expect(result.agent).toBe("CodeActAgent");

    // Nested shape derived for the local-mode settings page.
    expect(result.agent_settings?.agent).toBe("CodeActAgent");
    expect(result.agent_settings?.llm).toEqual({
      model: "anthropic/claude-3-5-sonnet",
      base_url: "https://api.anthropic.com",
    });
    expect(result.conversation_settings?.confirmation_mode).toBe(true);
    expect(result.conversation_settings?.security_analyzer).toBe("llm");
    expect(result.conversation_settings?.max_iterations).toBe(30);
  });

  it("derives every supported nested field while preserving cloud defaults", async () => {
    const flat = {
      llm_model: "openai/gpt-4o",
      llm_base_url: "https://api.openai.com",
      llm_api_key: "encrypted-api-key",
      llm_api_key_set: false,
      search_api_key_set: true,
      enable_default_condenser: false,
      condenser_max_size: 0,
      agent: "CodeActAgent",
      mcp_config: {
        calendar: { url: "https://calendar.example.com/mcp" },
      },
      confirmation_mode: false,
      security_analyzer: null,
      max_iterations: 0,
      extra_cloud_field: "preserved",
    };
    stubGet("/api/v1/settings", flat);

    const result = await fetchCloudSettings();

    expect(result).toStrictEqual({
      ...flat,
      agent_settings: {
        llm: {
          model: "openai/gpt-4o",
          base_url: "https://api.openai.com",
          api_key: "encrypted-api-key",
        },
        condenser: { enabled: false, max_size: 0 },
        agent: "CodeActAgent",
        mcp_config: flat.mcp_config,
      },
      conversation_settings: {
        confirmation_mode: false,
        security_analyzer: null,
        max_iterations: 0,
      },
      llm_api_key_set: false,
      search_api_key_set: true,
      provider_tokens_set: undefined,
    });
  });

  it("preserves non-empty nested settings instead of replacing them from flat fields", async () => {
    const flat = {
      llm_model: "flat-model",
      confirmation_mode: false,
      agent_settings: {
        llm: { model: "nested-model" },
        custom_agent_value: "kept",
      },
      conversation_settings: {
        confirmation_mode: true,
        custom_conversation_value: "kept",
      },
    };
    stubGet("/api/v1/settings", flat);

    const result = await fetchCloudSettings();

    // Non-empty nested blocks are preserved value-for-value (the derived
    // fallback only kicks in when they are absent/empty). Cross-realm fetch
    // parsing means these are structurally-equal clones, not the same refs.
    expect(result.agent_settings).toEqual(flat.agent_settings);
    expect(result.conversation_settings).toEqual(flat.conversation_settings);
  });

  it("derives empty nested settings and false key flags from a sparse response", async () => {
    const flat = {
      agent_settings: {},
      conversation_settings: {},
      mcp_config: {},
    };
    stubGet("/api/v1/settings", flat);

    const result = await fetchCloudSettings();

    expect(result).toStrictEqual({
      ...flat,
      agent_settings: {},
      conversation_settings: {},
      llm_api_key_set: false,
      search_api_key_set: false,
      provider_tokens_set: undefined,
    });
  });

  it("derives flat values when the cloud returns empty nested blocks", async () => {
    const flat = {
      agent_settings: {},
      conversation_settings: {},
      llm_model: "fallback-model",
      confirmation_mode: false,
    };
    stubGet("/api/v1/settings", flat);

    const result = await fetchCloudSettings();

    expect(result.agent_settings).toEqual({
      llm: { model: "fallback-model" },
    });
    expect(result.conversation_settings).toEqual({ confirmation_mode: false });
  });

  it("rejects before proxying when the active backend is local", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });

    const fetchSpy = vi.spyOn(globalThis, "fetch");
    try {
      await expect(fetchCloudSettings()).rejects.toThrow(
        "Cloud settings call requires a cloud backend.",
      );
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      fetchSpy.mockRestore();
    }
  });

  it("propagates cloud proxy failures unchanged", async () => {
    server.use(
      http.get(SETTINGS_URL, ({ request }) => {
        capturedRequests.push(request);
        return HttpResponse.error();
      }),
    );

    // A transport-level failure bubbles straight out of fetchCloudSettings
    // instead of being swallowed or turned into a resolved value.
    await expect(fetchCloudSettings()).rejects.toThrow();
    expect(capturedRequests).toHaveLength(1);
  });

  it("saveCloudSettings forwards diffs verbatim and omits the legacy keys the cloud rejects", async () => {
    stubPost("/api/v1/settings");

    const agentDiff = {
      llm: { model: "openai/gpt-4o", base_url: "https://api.openai.com" },
      agent: "CodeActAgent",
    };
    const conversationDiff = { max_iterations: 50 };

    await saveCloudSettings({
      agent_settings_diff: agentDiff,
      conversation_settings_diff: conversationDiff,
    });

    const request = capturedRequests[0]!;
    expect(request.method).toBe("POST");
    expect(request.url).toBe(SETTINGS_URL);
    expect(request.headers.get("authorization")).toBe("Bearer bearer-token");

    const requestBody = capturedBodies[0]!;
    expect(requestBody).toEqual({
      agent_settings_diff: agentDiff,
      conversation_settings_diff: conversationDiff,
    });
    expect(requestBody).not.toHaveProperty("agent_settings");
    expect(requestBody).not.toHaveProperty("conversation_settings");
  });

  it("SettingsService.saveSettings forwards disabled_skills to cloud when active backend is cloud", async () => {
    // Arrange: cloud backend already active via beforeEach; mock cloud response.
    stubPost("/api/v1/settings");

    // Act: save a skills-only update — previously this short-circuited and
    // sent nothing at all, leaving the toggle un-persisted.
    await SettingsService.saveSettings({
      disabled_skills: ["SSH Microagent"],
    });

    // Assert: a single POST /api/v1/settings reached the wire with
    // disabled_skills as a top-level field.
    expect(capturedRequests).toHaveLength(1);
    const request = capturedRequests[0]!;
    expect(request.method).toBe("POST");
    expect(request.url).toBe(SETTINGS_URL);
    expect(request.headers.get("authorization")).toBe("Bearer bearer-token");
    expect(capturedBodies[0]).toEqual({
      disabled_skills: ["SSH Microagent"],
    });
  });

  it("saveCloudSettings omits an empty conversation_settings_diff (LLM-only save)", async () => {
    stubPost("/api/v1/settings");

    await saveCloudSettings({
      agent_settings_diff: {
        llm: { model: "anthropic/claude-sonnet-4-20250514" },
      },
      conversation_settings_diff: {},
    });

    expect(capturedBodies[0]).toEqual({
      agent_settings_diff: {
        llm: { model: "anthropic/claude-sonnet-4-20250514" },
      },
    });
  });

  it("sends explicit preference clears while omitting undefined preferences", async () => {
    stubPost("/api/v1/settings");

    await saveCloudSettings({
      app_preferences: {
        language: undefined,
        user_consents_to_analytics: null,
        enable_sound_notifications: false,
        git_user_name: "",
        disabled_skills: [],
      },
    });

    const request = capturedRequests[0]!;
    expect(request.method).toBe("POST");
    expect(request.url).toBe(SETTINGS_URL);
    expect(request.headers.get("authorization")).toBe("Bearer bearer-token");
    expect(capturedBodies[0]).toEqual({
      user_consents_to_analytics: null,
      enable_sound_notifications: false,
      git_user_name: "",
      disabled_skills: [],
    });
    expect(capturedBodies[0]).not.toHaveProperty("language");
  });

  it("sends an empty payload when no settings changed", async () => {
    stubPost("/api/v1/settings");

    await saveCloudSettings({});

    const request = capturedRequests[0]!;
    expect(request.method).toBe("POST");
    expect(request.url).toBe(SETTINGS_URL);
    expect(request.headers.get("authorization")).toBe("Bearer bearer-token");
    expect(capturedBodies[0]).toEqual({});
  });

  it.each([
    ["agent", fetchCloudSettingsSchema, "/api/v1/settings/agent-schema"],
    [
      "conversation",
      fetchCloudConversationSettingsSchema,
      "/api/v1/settings/conversation-schema",
    ],
  ] as const)(
    "fetches the %s settings schema from the exact cloud route",
    async (_label, fetchSchema, path) => {
      const schema = { model_name: `${_label}-settings`, sections: [] };
      stubGet(path, schema);

      const result = await fetchSchema();

      // The parsed JSON round-trips value-for-value (fetch yields a fresh
      // object, so this is structural equality rather than reference identity).
      expect(result).toEqual(schema);
      const request = capturedRequests[0]!;
      expect(request.method).toBe("GET");
      expect(request.url).toBe(`${cloudBackend.host}${path}`);
      expect(request.headers.get("authorization")).toBe("Bearer bearer-token");
    },
  );
});

describe("saveCloudSettings drops agent_context: null (agent-canvas#981)", () => {
  it("strips a null agent_context while preserving sibling agent settings", async () => {
    // Arrange: the cloud rejects agent_context: null against OpenHandsAgentSettings.
    stubPost("/api/v1/settings");

    // Act
    await saveCloudSettings({
      agent_settings_diff: {
        llm: { model: "anthropic/claude-sonnet-4-20250514" },
        agent_context: null,
      },
    });

    // Assert: agent_context never reaches the wire, but the real llm change does.
    expect(capturedBodies[0]).toEqual({
      agent_settings_diff: {
        llm: { model: "anthropic/claude-sonnet-4-20250514" },
      },
    });
  });

  it("preserves a null mcp_config so clearing MCP servers still round-trips", async () => {
    // Arrange: mcp_config: null is an intentional "clear" signal, not an error.
    stubPost("/api/v1/settings");

    // Act
    await saveCloudSettings({
      agent_settings_diff: { mcp_config: null },
    });

    // Assert: the null mcp_config must survive (don't over-strip nulls).
    expect(capturedBodies[0]).toEqual({
      agent_settings_diff: { mcp_config: null },
    });
  });

  it("preserves a non-null agent_context", async () => {
    stubPost("/api/v1/settings");
    const agentContext = {
      system_message_suffix: "Keep this context",
    };

    await saveCloudSettings({
      agent_settings_diff: { agent_context: agentContext },
    });

    expect(capturedBodies[0]).toEqual({
      agent_settings_diff: { agent_context: agentContext },
    });
  });

  it("omits agent_settings_diff when agent_context: null is its only key", async () => {
    // Arrange
    stubPost("/api/v1/settings");

    // Act
    await saveCloudSettings({
      agent_settings_diff: { agent_context: null },
    });

    // Assert: nothing is left to send, so no agent_settings_diff goes on the wire.
    expect(capturedBodies[0]).toEqual({});
  });
});
