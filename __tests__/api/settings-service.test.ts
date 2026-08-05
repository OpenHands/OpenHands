import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { http, HttpResponse } from "msw";

import SettingsService from "#/api/settings-service/settings-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { server } from "#/mocks/node";
import { resetTestHandlersMockSettings } from "#/mocks/settings-handlers";
import type { Settings } from "#/types/settings";
import { buildMcpServerPatch } from "#/utils/mcp-config";

const mockSaveCloudSettings = vi.fn();
const mockFetchCloudSettings = vi.fn();

vi.mock("#/api/cloud/settings-service.api", () => ({
  saveCloudSettings: (args: unknown) => mockSaveCloudSettings(args),
  fetchCloudSettings: () => mockFetchCloudSettings(),
  fetchCloudSettingsSchema: vi.fn(),
  fetchCloudConversationSettingsSchema: vi.fn(),
}));

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

describe("SettingsService", () => {
  beforeEach(() => {
    // Clear localStorage and reset mock settings state
    window.localStorage.clear();
    resetTestHandlersMockSettings();
    __resetActiveStoreForTests();
    mockSaveCloudSettings.mockReset().mockResolvedValue(undefined);
    mockFetchCloudSettings.mockReset();
    // Invalidate the in-memory cache
    SettingsService.invalidateCache();
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("fetches settings from the API and normalizes derived fields", async () => {
    // The mock handler returns default settings
    const settings = await SettingsService.getSettings();

    // Should have normalized settings with derived fields
    expect(settings.agent).toBe("CodeActAgent");
    expect(settings.llm_model).toBe("openhands/glm-5.2");
    expect(settings.confirmation_mode).toBe(false);
    expect(settings.security_analyzer).toBe("llm");
  });

  it("saves settings via PATCH API and invalidates cache", async () => {
    // Save some settings
    await SettingsService.saveSettings({
      agent_settings_diff: {
        agent: "CodeActAgent",
        llm: {
          model: "saved-model",
          base_url: "https://saved.example.com",
          api_key: "saved-key",
        },
      },
      conversation_settings_diff: {
        confirmation_mode: true,
        security_analyzer: "llm",
        max_iterations: 33,
      },
    });

    // Fetch settings again - should reflect the saved values
    const settings = await SettingsService.getSettings();

    expect(settings.llm_model).toBe("saved-model");
    expect(settings.llm_base_url).toBe("https://saved.example.com");
    // Note: api_key will be redacted when fetched without X-Expose-Secrets header
    expect(settings.confirmation_mode).toBe(true);
    expect(settings.security_analyzer).toBe("llm");
    expect(settings.max_iterations).toBe(33);
  });

  it("returns encrypted secrets when using getSettingsForConversation", async () => {
    // First save a key
    await SettingsService.saveSettings({
      agent_settings_diff: {
        llm: {
          api_key: "test-api-key",
        },
      },
    });

    // Get settings for conversation (should have encrypted secrets)
    const { agentSettings, secretsEncrypted } =
      await SettingsService.getSettingsForConversation();

    expect(secretsEncrypted).toBe(true);
    // The mock returns an "encrypted" placeholder for the key
    const llm = agentSettings.llm as Record<string, unknown> | undefined;
    expect(llm?.api_key).toMatch(/^gAAAAA_mock_encrypted_/);
  });

  it("uses cache for repeated getSettings calls", async () => {
    const fetchSpy = vi.spyOn(SettingsService, "fetchSettingsFromApi");

    // First call - should fetch from API
    await SettingsService.getSettings();
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    // Second call - should use cache
    await SettingsService.getSettings();
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    // After invalidation - should fetch again
    SettingsService.invalidateCache();
    await SettingsService.getSettings();
    expect(fetchSpy).toHaveBeenCalledTimes(2);

    fetchSpy.mockRestore();
  });

  it("skips API call when no diffs are provided to saveSettings", async () => {
    const fetchSpy = vi.spyOn(SettingsService, "fetchSettingsFromApi");

    // Call with empty/no diffs
    const result = await SettingsService.saveSettings({});

    expect(result).toBe(true);
    // No fetch should have been made (PATCH not called)
    expect(fetchSpy).not.toHaveBeenCalled();

    fetchSpy.mockRestore();
  });

  it("sends disabled_skills under misc_settings_diff.app_preferences on a local backend", async () => {
    // disabled_skills is one of the AppPreferences fields (along with
    // language, git identity, …) and is persisted server-side under
    // `PersistedSettings.misc_settings.app_preferences` since the
    // misc_settings container was introduced as a follow-up to PR #3539.
    const patchBodies: Array<Record<string, unknown>> = [];
    server.use(
      http.patch("*/api/settings", async ({ request }) => {
        patchBodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
          misc_settings: {
            app_preferences: { disabled_skills: ["SSH Microagent"] },
          },
        });
      }),
    );

    const result = await SettingsService.saveSettings({
      disabled_skills: ["SSH Microagent"],
    });

    expect(result).toBe(true);
    expect(patchBodies).toEqual([
      {
        misc_settings_diff: {
          app_preferences: { disabled_skills: ["SSH Microagent"] },
        },
      },
    ]);
  });

  it("stores the title profile as a local app preference", async () => {
    const patchBodies: Array<Record<string, unknown>> = [];
    server.use(
      http.patch("*/api/settings", async ({ request }) => {
        patchBodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
          misc_settings: {
            app_preferences: { title_llm_profile: "Titles" },
          },
        });
      }),
    );

    await SettingsService.saveSettings({ title_llm_profile: "Titles" });

    expect(patchBodies).toEqual([
      {
        misc_settings_diff: {
          app_preferences: { title_llm_profile: "Titles" },
        },
      },
    ]);
  });

  it("treats omitted app-preferences analytics consent as unset", async () => {
    server.use(
      http.get("*/api/settings", () =>
        HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
          misc_settings: { app_preferences: {} },
        }),
      ),
    );
    SettingsService.invalidateCache();

    const settings = await SettingsService.getSettings();

    expect(settings.user_consents_to_analytics).toBeNull();
  });

  it("surfaces server-side misc_settings.app_preferences in getSettings on a local backend", async () => {
    // The local agent-server returns app_preferences nested under
    // `misc_settings` on GET /api/settings. The mock handler echoes whatever
    // was last PATCH'd; seed it through the real save path so the round-trip
    // matches production.
    const appPrefs = {
      language: "fr",
      git_user_name: "Alice",
      git_user_email: "alice@example.com",
      enable_sound_notifications: true,
      user_consents_to_analytics: true,
      title_llm_profile: "Titles",
      disabled_skills: ["SSH Microagent"],
    };
    await SettingsService.saveSettings(appPrefs);
    SettingsService.invalidateCache();

    const settings = await SettingsService.getSettings();

    expect({
      language: settings.language,
      git_user_name: settings.git_user_name,
      git_user_email: settings.git_user_email,
      enable_sound_notifications: settings.enable_sound_notifications,
      user_consents_to_analytics: settings.user_consents_to_analytics,
      title_llm_profile: settings.title_llm_profile,
      disabled_skills: settings.disabled_skills,
    }).toEqual(appPrefs);
  });

  it("routes app-level fields into misc_settings_diff when mixed with agent diffs", async () => {
    const patchBodies: Array<Record<string, unknown>> = [];
    server.use(
      http.patch("*/api/settings", async ({ request }) => {
        patchBodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          agent_settings: { agent: "CodeActAgent" },
          conversation_settings: {},
          llm_api_key_is_set: false,
          misc_settings: { app_preferences: { git_user_name: "Alice" } },
        });
      }),
    );

    await SettingsService.saveSettings({
      git_user_name: "Alice",
      agent_settings_diff: { agent: "CodeActAgent" },
    });

    expect(patchBodies).toEqual([
      {
        agent_settings_diff: { agent: "CodeActAgent" },
        misc_settings_diff: { app_preferences: { git_user_name: "Alice" } },
      },
    ]);
  });

  it("forwards app-level preferences as flat top-level fields to the cloud save", async () => {
    // Arrange: switch the active backend to cloud so saveSettings routes
    // through saveCloudSettings (mocked).
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    // Act
    await SettingsService.saveSettings({
      language: "fr",
      git_user_name: "Alice",
      title_llm_profile: null,
    });

    // Assert: cloud save received the fields under `app_preferences` so
    // `saveCloudSettings` can spread them into the POST body as flat keys.
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      app_preferences: {
        language: "fr",
        git_user_name: "Alice",
        title_llm_profile: null,
      },
    });
  });

  it("ignores any stale localStorage git-provider-tokens key (PAT layer removed)", async () => {
    // Arrange: a previous version of the app may have written PATs to
    // localStorage under this key. After removing the integrations page and
    // the PAT layer, getSettings must not resurrect those stale tokens into
    // provider_tokens_set — that would re-enable the removed flow.
    window.localStorage.setItem(
      "openhands-agent-server-git-provider-tokens",
      JSON.stringify({
        github: { token: "ghp_stale_xyz", host: "github.com" },
        gitlab: { token: "glpat_stale_xyz", host: null },
      }),
    );

    // Act
    const settings = await SettingsService.getSettings();

    // Assert
    expect(settings.provider_tokens_set).toEqual({});
  });

  it("sends an mcp_config merge-patch in one request on the local backend", async () => {
    const patchBodies: Array<Record<string, unknown>> = [];
    server.use(
      http.patch("*/api/settings", async ({ request }) => {
        patchBodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
        });
      }),
    );

    await SettingsService.saveSettings({
      agent_settings_diff: {
        mcp_config: { only: { url: "https://x.example" } },
      },
    });

    expect(patchBodies).toEqual([
      {
        agent_settings_diff: {
          mcp_config: { only: { url: "https://x.example" } },
        },
      },
    ]);
  });

  it("creates, patches, and deletes one named MCP entry with one local request each", async () => {
    const requests: Array<{
      method: string;
      settingsKey: string;
      body?: Record<string, unknown>;
    }> = [];
    const response = {
      agent_settings: {},
      conversation_settings: {},
      llm_api_key_is_set: false,
    };
    server.use(
      http.post(
        "*/api/settings/mcp/:settingsKey",
        async ({ params, request }) => {
          requests.push({
            method: "POST",
            settingsKey: String(params.settingsKey),
            body: (await request.json()) as Record<string, unknown>,
          });
          return HttpResponse.json(response, { status: 201 });
        },
      ),
      http.patch(
        "*/api/settings/mcp/:settingsKey",
        async ({ params, request }) => {
          requests.push({
            method: "PATCH",
            settingsKey: String(params.settingsKey),
            body: (await request.json()) as Record<string, unknown>,
          });
          return HttpResponse.json(response);
        },
      ),
      http.delete("*/api/settings/mcp/:settingsKey", ({ params }) => {
        requests.push({
          method: "DELETE",
          settingsKey: String(params.settingsKey),
        });
        return HttpResponse.json(response);
      }),
    );

    await SettingsService.createMcpServer("docs", {
      transport: "http",
      url: "https://docs.example/mcp",
    });
    await SettingsService.patchMcpServer("github", {
      url: "https://github.example/v2/mcp",
    });
    await SettingsService.deleteMcpServer("old");

    expect(requests).toEqual([
      {
        method: "POST",
        settingsKey: "docs",
        body: {
          transport: "http",
          url: "https://docs.example/mcp",
        },
      },
      {
        method: "PATCH",
        settingsKey: "github",
        body: { url: "https://github.example/v2/mcp" },
      },
      {
        method: "DELETE",
        settingsKey: "old",
      },
    ]);
  });

  it("sends auth replacement tombstones to the named MCP endpoint", async () => {
    const patchBodies: Array<Record<string, unknown>> = [];
    server.use(
      http.patch("*/api/settings/mcp/:settingsKey", async ({ request }) => {
        patchBodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
        });
      }),
    );

    await SettingsService.patchMcpServer(
      "mail",
      buildMcpServerPatch(
        {
          transport: "http",
          url: "https://mail.example/mcp",
          auth: {
            strategy: "oauth2",
            authentication: { type: "oauth", scopes: "mail.read" },
            state: { tokens: { access_token: "**********" } },
          },
        },
        {
          id: "mail",
          type: "shttp",
          name: "mail",
          url: "https://mail.example/mcp",
          auth: { strategy: "bearer", value: "replacement-token" },
        },
      ),
    );

    expect(patchBodies).toEqual([
      {
        transport: "http",
        url: "https://mail.example/mcp",
        auth: {
          strategy: "bearer",
          value: "replacement-token",
          authentication: null,
          state: null,
        },
      },
    ]);
  });

  it("does not pre-clear when the mcp_config diff is already null on the local backend", async () => {
    // When the caller is wiping mcp_config entirely (e.g. user removed the
    // last server), a single PATCH already takes effect because null is
    // not a dict and deep-merge replaces rather than recurses. A second
    // clear would be wasted work.
    const patchBodies: Array<Record<string, unknown>> = [];
    server.use(
      http.patch("*/api/settings", async ({ request }) => {
        patchBodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
        });
      }),
    );

    await SettingsService.saveSettings({
      agent_settings_diff: { mcp_config: null },
    });

    expect(patchBodies).toEqual([
      { agent_settings_diff: { mcp_config: null } },
    ]);
  });

  it("does not pre-clear when the diff has no mcp_config on the local backend", async () => {
    // A typical settings save (LLM model, condenser, …) must NOT incur the
    // mcp_config pre-clear round-trip — that would needlessly drop the
    // user's MCP servers if anything ever raced.
    const patchBodies: Array<Record<string, unknown>> = [];
    server.use(
      http.patch("*/api/settings", async ({ request }) => {
        patchBodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
        });
      }),
    );

    await SettingsService.saveSettings({
      agent_settings_diff: { agent: "CodeActAgent" },
    });

    expect(patchBodies).toEqual([
      { agent_settings_diff: { agent: "CodeActAgent" } },
    ]);
  });

  it("sends an mcp_config merge-patch in one request on the cloud backend", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await SettingsService.saveSettings({
      agent_settings_diff: {
        mcp_config: { only: { url: "https://x.example" } },
      },
    });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: { only: { url: "https://x.example" } },
      },
    });
  });

  it("does not pre-clear cloud mcp_config when rotating a redacted MCP key", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    mockFetchCloudSettings.mockResolvedValue({
      agent_settings: {
        mcp_config: {
          integrations_hub: {
            url: "https://integrations.staging.all-hands.dev/api/mcp",
            headers: { Authorization: "**********" },
          },
        },
      },
    });

    await SettingsService.saveSettings({
      agent_settings_diff: {
        mcp_config: {
          integrations_hub: {
            url: "https://integrations.staging.all-hands.dev/api/mcp",
            auth: { strategy: "bearer", value: "new-key" },
          },
        },
      },
    });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: {
          integrations_hub: {
            url: "https://integrations.staging.all-hands.dev/api/mcp",
            headers: { Authorization: "Bearer new-key" },
          },
        },
      },
    });
  });

  it("converts bearer MCP auth to headers when saving mcp_config to cloud", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await SettingsService.saveSettings({
      agent_settings_diff: {
        mcp_config: {
          elevenlabs: {
            transport: "http",
            url: "https://mcp.example.com/mcp",
            auth: { strategy: "bearer", value: "elevenlabs-test-token" },
          },
        },
      },
    });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: {
          elevenlabs: {
            transport: "http",
            url: "https://mcp.example.com/mcp",
            headers: {
              Authorization: "Bearer elevenlabs-test-token",
            },
          },
        },
      },
    });
  });

  it("converts OAuth token state to headers when saving mcp_config to cloud", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await SettingsService.saveSettings({
      agent_settings_diff: {
        mcp_config: {
          notion: {
            transport: "http",
            url: "https://mcp.example.com/mcp",
            auth: {
              strategy: "oauth2",
              authentication: {
                type: "oauth",
                client_auth_method: "client_secret_post",
              },
              state: {
                tokens: {
                  access_token: "oauth-access-token",
                },
              },
            },
          },
        },
      },
    });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: {
          notion: {
            transport: "http",
            url: "https://mcp.example.com/mcp",
            headers: {
              Authorization: "Bearer oauth-access-token",
            },
          },
        },
      },
    });
  });

  it("clears cloud MCP credential headers when auth is explicitly cleared", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await SettingsService.patchMcpServer("github", { auth: null });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: {
          github: { headers: null },
        },
      },
    });
  });

  it("tombstones the old credential header when a cloud MCP switches auth strategy", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    mockFetchCloudSettings.mockResolvedValue({
      mcp_config: {
        github: {
          url: "https://github.example/mcp",
          headers: { "X-API-Key": "stale-custom-header-secret" },
        },
      },
    });

    await SettingsService.patchMcpServer("github", {
      auth: { strategy: "bearer", value: "new-bearer-token" },
    });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: {
          github: {
            headers: {
              Authorization: "Bearer new-bearer-token",
              "X-API-Key": null,
            },
          },
        },
      },
    });
  });

  it("keeps stored non-credential headers when a cloud MCP switches auth strategy", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    mockFetchCloudSettings.mockResolvedValue({
      mcp_config: {
        github: {
          url: "https://github.example/mcp",
          headers: {
            "X-API-Key": "stale-custom-header-secret",
            "X-Trace": "on",
          },
        },
      },
    });

    await SettingsService.patchMcpServer("github", {
      auth: { strategy: "bearer", value: "new-bearer-token" },
      headers: { "X-Trace": "on" },
    });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: {
          github: {
            headers: {
              Authorization: "Bearer new-bearer-token",
              "X-Trace": "on",
              "X-API-Key": null,
            },
          },
        },
      },
    });
  });

  it("does not fetch stored cloud settings for a patch with no auth credential", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await SettingsService.patchMcpConfig({
      only: { url: "https://x.example" },
    });

    expect(mockFetchCloudSettings).not.toHaveBeenCalled();
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: { only: { url: "https://x.example" } },
      },
    });
  });

  it("tombstones stored credential headers when a cloud MCP auth becomes none", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    mockFetchCloudSettings.mockResolvedValue({
      mcp_config: {
        github: {
          url: "https://github.example/mcp",
          headers: { Authorization: "Bearer stale-token" },
        },
      },
    });

    await SettingsService.patchMcpServer("github", {
      auth: { strategy: "none" },
    });

    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(1);
    expect(mockSaveCloudSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        mcp_config: {
          github: {
            headers: { Authorization: null },
          },
        },
      },
    });
  });

  it("does not mutate the prior cloud catalog before a failed merge-patch", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    const previousMcpConfig = { existing: { url: "https://old.example" } };
    mockFetchCloudSettings.mockResolvedValue({
      agent_settings: { mcp_config: previousMcpConfig },
    });

    mockSaveCloudSettings.mockImplementation(
      (args: { agent_settings_diff?: { mcp_config?: unknown } }) => {
        const mcp = args?.agent_settings_diff?.mcp_config;
        if (
          mcp &&
          typeof mcp === "object" &&
          (mcp as Record<string, unknown>).new
        ) {
          return Promise.reject(new Error("validation failed"));
        }
        return Promise.resolve(undefined);
      },
    );

    await expect(
      SettingsService.saveSettings({
        agent_settings_diff: {
          mcp_config: { new: { url: "https://new.example" } },
        },
      }),
    ).rejects.toThrow("validation failed");

    expect(mockFetchCloudSettings).not.toHaveBeenCalled();
    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(3);
    expect(mockSaveCloudSettings.mock.calls).not.toContainEqual([
      { agent_settings_diff: { mcp_config: null } },
    ]);
  });

  it("does not mutate the prior local catalog before a failed merge-patch", async () => {
    const patchBodies: Array<Record<string, unknown>> = [];
    const previousMcpConfig = {
      existing: { url: "https://old.example" },
    };
    let getCount = 0;
    server.use(
      http.get("*/api/settings", () => {
        getCount += 1;
        return HttpResponse.json({
          agent_settings: { mcp_config: previousMcpConfig },
          conversation_settings: {},
          llm_api_key_is_set: false,
        });
      }),
      http.patch("*/api/settings", async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        patchBodies.push(body);
        const agentDiff = body.agent_settings_diff as
          | { mcp_config?: unknown }
          | undefined;
        const mcp = agentDiff?.mcp_config;
        if (
          mcp &&
          typeof mcp === "object" &&
          (mcp as Record<string, unknown>).new
        ) {
          return HttpResponse.json(
            { error: "validation failed" },
            { status: 400 },
          );
        }
        return HttpResponse.json({
          agent_settings: {},
          conversation_settings: {},
          llm_api_key_is_set: false,
        });
      }),
    );

    await expect(
      SettingsService.saveSettings({
        agent_settings_diff: {
          mcp_config: { new: { url: "https://new.example" } },
        },
      }),
    ).rejects.toBeDefined();

    expect(getCount).toBe(0);
    expect(patchBodies).toHaveLength(3);
    expect(patchBodies).not.toContainEqual({
      agent_settings_diff: { mcp_config: null },
    });
  });

  it("does not fetch a snapshot or issue a destructive clear when a cloud patch fails", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    mockFetchCloudSettings.mockResolvedValue({ agent_settings: {} });
    mockSaveCloudSettings.mockImplementation(
      (_args: { agent_settings_diff?: { mcp_config?: unknown } }) => {
        return Promise.reject(new Error("validation failed"));
      },
    );

    await expect(
      SettingsService.saveSettings({
        agent_settings_diff: {
          mcp_config: { new: { url: "https://new.example" } },
        },
      }),
    ).rejects.toThrow("validation failed");

    expect(mockFetchCloudSettings).not.toHaveBeenCalled();
    expect(mockSaveCloudSettings).toHaveBeenCalledTimes(3);
    const sentMcpValues = mockSaveCloudSettings.mock.calls.map(
      (call) =>
        (call[0] as { agent_settings_diff?: { mcp_config?: unknown } })
          ?.agent_settings_diff?.mcp_config,
    );
    for (const mcp of sentMcpValues) {
      const isNewWrite =
        !!mcp &&
        typeof mcp === "object" &&
        !!(mcp as Record<string, unknown>).new;
      expect(isNewWrite).toBe(true);
    }
  });

  it("surfaces cloud app preferences on getSettings", async () => {
    // The cloud returns app-preference fields flat at the top level
    // (language, git identity, …) — they should land on the returned
    // Settings unchanged.
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    mockFetchCloudSettings.mockResolvedValue({
      language: "ja",
    } as Partial<Settings>);

    const settings = await SettingsService.getSettings();

    expect(settings.language).toBe("ja");
  });
});
