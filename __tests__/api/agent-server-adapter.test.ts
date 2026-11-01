import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockInstance } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import { CANVAS_UI_CLIENT_TOOL_NAME } from "#/constants/canvas-ui";
import { LAUNCH_CHILD_CONVERSATION_TOOL_NAME } from "#/constants/child-conversation";
import { REQUEST_SECRET_TOOL_NAME } from "#/constants/secret-request";
import { AUTOMATION_FORM_UPDATE_TOOL_NAME } from "#/constants/automation-form";

import {
  ACP_SERVER_TAG_KEY,
  WORKSPACE_TAG_KEY,
  buildWorkspaceTags,
  buildRuntimeServicesSystemSuffix,
  buildStartConversationRequest,
  buildStartConversationRequestWithEncryptedSettings,
  fetchBackendExecutionRuntime,
  getDefaultConversationTitle,
  parseRuntimeServicesInfo,
  toAppConversation,
  toHooksResponse,
  type DirectConversationInfo,
} from "#/api/agent-server-adapter";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SecretsService } from "#/api/secrets-service";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  removeStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { HookType } from "@openhands/typescript-client";
import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import type { HookConfig } from "@openhands/typescript-client";
import { DEFAULT_SETTINGS } from "#/services/settings";
import type { SettingsValue } from "#/types/settings";
import { useModelStore } from "#/stores/model-store";

const {
  mockGetAgentServerWorkingDir,
  mockGetEffectiveLocalBackend,
  mockGetCachedAgentServerInfo,
  mockGetServerInfo,
  mockGetActiveBackend,
  mockLoadHooks,
} = vi.hoisted(() => ({
  mockGetAgentServerWorkingDir: vi.fn(() => "/workspace/project/agent-canvas"),
  mockGetEffectiveLocalBackend: vi.fn(() => ({
    id: "default-local",
    name: "Local backend",
    host: "http://127.0.0.1:8000",
    apiKey: "session-key",
    kind: "local" as const,
  })),
  mockGetCachedAgentServerInfo: vi.fn<() => unknown>(() => null),
  mockGetServerInfo: vi.fn(),
  mockGetActiveBackend: vi.fn(() => ({
    backend: {
      id: "default-local",
      name: "Local backend",
      host: "http://127.0.0.1:8000",
      apiKey: "session-key",
      kind: "local" as const,
    },
  })),
  mockLoadHooks: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(function ServerClientMock() {
    return {
      getServerInfo: mockGetServerInfo,
    };
  }),
  HooksClient: vi.fn(function HooksClientMock() {
    return { loadHooks: mockLoadHooks };
  }),
}));

vi.mock("#/api/agent-server-config", () => ({
  getAgentServerBaseUrl: vi.fn(() => "http://127.0.0.1:8000"),
  getAgentServerSessionApiKey: vi.fn(() => null),
  getAgentServerWorkingDir: mockGetAgentServerWorkingDir,
  shouldLoadPublicSkills: vi.fn(() => true),
  syncBakedSessionApiKey: vi.fn(),
}));

vi.mock("#/api/agent-server-compatibility", () => ({
  getCachedAgentServerInfo: mockGetCachedAgentServerInfo,
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getEffectiveLocalBackend: mockGetEffectiveLocalBackend,
  getActiveBackend: mockGetActiveBackend,
  isNoBackend: (backend: { id: string }) => backend.id === "no-backend",
}));

// `HookConfig` requires every event key, so build fixtures from a complete base.
const makeHookConfig = (overrides: Partial<HookConfig> = {}): HookConfig => ({
  pre_tool_use: [],
  post_tool_use: [],
  user_prompt_submit: [],
  session_start: [],
  session_end: [],
  stop: [],
  ...overrides,
});

const WORKSPACE_HOOK_CONFIG = makeHookConfig({
  session_start: [
    {
      matcher: "*",
      hooks: [{ command: "cat AGENTS.md", type: HookType.COMMAND }],
    },
  ],
});

const EXPLICIT_HOOK_CONFIG = makeHookConfig({
  session_start: [
    {
      matcher: "*",
      hooks: [{ command: "echo explicit", type: HookType.COMMAND }],
    },
  ],
});

beforeEach(() => {
  mockGetCachedAgentServerInfo.mockReturnValue(null);
  mockGetServerInfo.mockReset();
  mockGetEffectiveLocalBackend.mockReturnValue({
    id: "default-local",
    name: "Local backend",
    host: "http://127.0.0.1:8000",
    apiKey: "session-key",
    kind: "local",
  });
  useModelStore.getState().setHomeManualModelPick(false);
});

const PROFILE = {
  agentProfileId: "profile-1",
  agentProfileKind: "openhands" as const,
};

describe("buildStartConversationRequest", () => {
  it.each([{ city: "Tokyo" }, { metric: false }, { days: 0 }])(
    "rejects unsupported plugin parameters instead of silently stripping %o",
    (parameters) => {
      expect(() =>
        buildStartConversationRequest({
          ...PROFILE,
          settings: DEFAULT_SETTINGS,
          plugins: [{ source: "local-plugin", parameters }],
        }),
      ).toThrow();
    },
  );

  it.each(
    [5, 0, true, false, "", "Tokyo", [], [false]].map((parameters) => ({
      parameters,
    })),
  )(
    "rejects malformed plugin parameters instead of silently stripping $parameters",
    ({ parameters }) => {
      // Launch links can carry JSON values outside PluginSpec's static type.
      const plugins = JSON.parse(
        JSON.stringify([{ source: "local-plugin", parameters }]),
      );
      expect(() =>
        buildStartConversationRequest({
          ...PROFILE,
          settings: DEFAULT_SETTINGS,
          plugins,
        }),
      ).toThrow();
    },
  );

  it.each([undefined, null, {}])(
    "keeps coordinate-only plugins when parameters are %o",
    (parameters) => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: DEFAULT_SETTINGS,
        plugins: [
          {
            source: "local-plugin",
            ref: "main",
            repo_path: "plugin",
            parameters,
          },
        ],
      });
      expect(payload.plugins).toEqual([
        { source: "local-plugin", ref: "main", repo_path: "plugin" },
      ]);
    },
  );

  it("includes the resolved title LLM profile when configured", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      titleLlmProfile: "Titles",
    });

    expect(payload.title_llm_profile).toBe("Titles");
  });

  it("sends no runtime options while Settings → Runtime is at its defaults", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        conversation_settings: {
          ...DEFAULT_SETTINGS.conversation_settings,
          worktree_base: "default_branch",
          worktree_base_ref: null,
          worktree_branch_prefix: "openhands/",
          container_memory: null,
          container_cpus: null,
          container_browser: true,
        },
      },
    });

    expect(payload).not.toHaveProperty("worktree_options");
    expect(payload).not.toHaveProperty("container");
  });

  it("sends the Settings → Runtime options a user changed", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        conversation_settings: {
          ...DEFAULT_SETTINGS.conversation_settings,
          worktree_base: "ref",
          worktree_base_ref: "origin/release",
          worktree_branch_prefix: "me/",
          container_memory: "4g",
          container_cpus: 2,
          container_browser: false,
        },
      },
    });

    expect(payload).toMatchObject({
      worktree_options: {
        base: "ref",
        ref: "origin/release",
        branch_prefix: "me/",
      },
      container: { memory: "4g", cpus: 2, browser: false },
    });
  });

  it("omits the title LLM profile in automatic mode without a default", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
    });

    expect(payload).not.toHaveProperty("title_llm_profile");
  });

  it("derives confirmation and security settings the same way as OpenHands", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "nested-model" },
        },
        conversation_settings: {
          ...DEFAULT_SETTINGS.conversation_settings,
          confirmation_mode: true,
          security_analyzer: "llm",
        },
      },
    }) as {
      confirmation_policy: Record<string, unknown>;
      security_analyzer: Record<string, unknown>;
    };

    expect(payload.confirmation_policy).toEqual({
      kind: "ConfirmRisky",
      threshold: "HIGH",
      confirm_unknown: true,
    });
    expect(payload.security_analyzer).toEqual({
      kind: "LLMSecurityAnalyzer",
    });
  });

  it("uses the supplied conversationId and workingDir overrides", () => {
    const conversationId = "11111111-1111-4111-8111-111111111111";
    const workingDir = `/base/${conversationId}`;
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "nested-model" },
        },
      },
      conversationId,
      workingDir,
    }) as {
      conversation_id?: string;
      workspace: { working_dir: string };
    };

    expect(payload.conversation_id).toBe(conversationId);
    expect(payload.workspace.working_dir).toBe(workingDir);
  });

  it("requests a git worktree for new conversations by default", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "nested-model" },
        },
      },
    }) as { worktree: boolean };

    expect(payload.worktree).toBe(true);
  });

  it("can start a conversation without requesting a git worktree", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "nested-model" },
        },
      },
      workingDir: "/Users/devin/project-without-git",
      worktree: false,
    }) as { worktree: boolean; workspace: { working_dir: string } };

    expect(payload.worktree).toBe(false);
    expect(payload.workspace.working_dir).toBe(
      "/Users/devin/project-without-git",
    );
  });

  it("forwards supported conversation runtime fields from nested settings", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "nested-model" },
        },
        conversation_settings: {
          ...DEFAULT_SETTINGS.conversation_settings,
          hook_config: { on_start: [] },
          tool_module_qualnames: { demo_tool: "pkg.tools.demo" },
          agent_definitions: [
            { name: "reviewer", system_prompt: "be helpful" },
          ],
        },
      },
      conversationInstructions: "Follow the repo conventions.",
      plugins: [
        { source: "github.com/org/plugin", ref: "main", repo_path: "/" },
      ],
    }) as Record<string, unknown>;

    expect(payload.hook_config).toEqual({ on_start: [] });
    expect(payload.tool_module_qualnames).toEqual({
      demo_tool: "pkg.tools.demo",
    });
    expect(payload.agent_definitions).toEqual([
      { name: "reviewer", system_prompt: "be helpful" },
    ]);
    expect(payload.plugins).toEqual([
      { source: "github.com/org/plugin", ref: "main", repo_path: "/" },
    ]);
    expect(payload.initial_message).toEqual({
      role: "user",
      content: [{ type: "text", text: "Follow the repo conventions." }],
      run: true,
    });
  });

  it("uses workspaceHookConfig when conversation_settings.hook_config is omitted", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      workspaceHookConfig: WORKSPACE_HOOK_CONFIG,
    }) as Record<string, unknown>;

    expect(payload.hook_config).toEqual(WORKSPACE_HOOK_CONFIG);
  });

  it("prioritizes conversation_settings.hook_config over workspaceHookConfig", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        conversation_settings: {
          ...DEFAULT_SETTINGS.conversation_settings,
          hook_config: EXPLICIT_HOOK_CONFIG as unknown as SettingsValue,
        },
      },
      workspaceHookConfig: WORKSPACE_HOOK_CONFIG,
    }) as Record<string, unknown>;

    expect(payload.hook_config).toEqual(EXPLICIT_HOOK_CONFIG);
  });

  it("serializes custom secrets as host-relative LookupSecret entries", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "nested-model" },
        },
      },
      customSecrets: [
        { name: "API_KEY", description: "Primary API key" },
        { name: "folder/name", description: "Nested secret" },
      ],
    }) as {
      secrets: Record<
        string,
        {
          kind: string;
          url: string;
          description?: string;
          headers?: Record<string, string>;
        }
      >;
    };

    expect(payload.secrets).toEqual({
      API_KEY: {
        kind: "LookupSecret",
        url: "/api/settings/secrets/API_KEY",
        description: "Primary API key",
        headers: { "X-Session-API-Key": "session-key" },
      },
      "folder/name": {
        kind: "LookupSecret",
        url: "/api/settings/secrets/folder%2Fname",
        description: "Nested secret",
        headers: { "X-Session-API-Key": "session-key" },
      },
    });
  });

  describe("secret delivery", () => {
    it("delivers provider credentials and user secrets as LookupSecrets for an ACP profile", () => {
      // Provider credentials (e.g. a saved CODEX_AUTH_JSON) ride as
      // LookupSecrets like any custom secret; the server keeps only the ones
      // the profile's `secret_refs` allows.
      const payload = buildStartConversationRequest({
        settings: DEFAULT_SETTINGS,
        agentProfileId: "profile-acp",
        agentProfileKind: "acp",
        customSecrets: [{ name: "CODEX_AUTH_JSON" }, { name: "MY_TOKEN" }],
      }) as unknown as { secrets: Record<string, { kind: string }> };

      expect(payload.secrets.CODEX_AUTH_JSON?.kind).toBe("LookupSecret");
      expect(payload.secrets.MY_TOKEN?.kind).toBe("LookupSecret");
    });

    it("never asks the server to decrypt request secrets", () => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: DEFAULT_SETTINGS,
        customSecrets: [{ name: "MY_TOKEN" }],
      });

      expect(payload).not.toHaveProperty("secrets_encrypted");
      expect(payload).not.toHaveProperty("agent_settings");
    });
  });

  describe("client tool injection", () => {
    it("sends canvas_ui as a client-defined JSON tool", () => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: DEFAULT_SETTINGS,
      });

      expect(payload.client_tools[0]).toMatchObject({
        name: CANVAS_UI_CLIENT_TOOL_NAME,
        parameters: {
          type: "object",
          properties: {
            command: {
              enum: ["navigate_to_file", "open_tab", "show_preview"],
            },
          },
          required: ["command"],
        },
        annotations: {
          readOnlyHint: true,
          destructiveHint: false,
          idempotentHint: true,
          openWorldHint: false,
        },
      });
      expect(payload.tool_module_qualnames).toBeUndefined();
    });

    it("omits the client tool for an ACP profile when global settings are stale", () => {
      const payload = buildStartConversationRequest({
        settings: DEFAULT_SETTINGS,
        agentProfileId: "profile-acp",
        agentProfileKind: "acp",
      });

      expect(payload.client_tools).toEqual([]);
    });

    it("sends the client tool for an OpenHands profile", () => {
      const payload = buildStartConversationRequest({
        settings: DEFAULT_SETTINGS,
        agentProfileId: "profile-openhands",
        agentProfileKind: "openhands",
      });

      expect(payload.client_tools.map((tool) => tool.name)).toEqual([
        CANVAS_UI_CLIENT_TOOL_NAME,
        LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
        REQUEST_SECRET_TOOL_NAME,
      ]);
    });

    it("adds automation setup tags to local conversation requests", () => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: DEFAULT_SETTINGS,
        automationSetup: true,
        automationSetupTags: {
          automationdraftid: "draft-1",
          automationmaterializeddraftid: "auto-1",
        },
      });

      expect(payload.tags).toEqual(
        expect.objectContaining({
          automationsetup: "draft",
          automationdraftid: "draft-1",
          automationmaterializeddraftid: "auto-1",
          clientsource: "agentcanvas",
        }),
      );
    });

    it.each([
      {
        automationSetup: false,
        expectedTools: [
          CANVAS_UI_CLIENT_TOOL_NAME,
          LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
          REQUEST_SECRET_TOOL_NAME,
        ],
      },
      {
        automationSetup: true,
        expectedTools: [
          CANVAS_UI_CLIENT_TOOL_NAME,
          LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
          REQUEST_SECRET_TOOL_NAME,
          AUTOMATION_FORM_UPDATE_TOOL_NAME,
        ],
      },
    ])(
      "builds client tools for automationSetup=$automationSetup",
      ({ automationSetup, expectedTools }) => {
        const payload = buildStartConversationRequest({
          ...PROFILE,
          settings: DEFAULT_SETTINGS,
          automationSetup,
        });

        expect(payload.client_tools.map((tool) => tool.name)).toEqual(
          expectedTools,
        );
      },
    );

    it("sends the client tool when resuming a conversation", () => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: DEFAULT_SETTINGS,
        conversationId: "legacy-conversation-id",
      }) as { conversation_id: string; client_tools: Array<{ name: string }> };

      expect(payload.conversation_id).toBe("legacy-conversation-id");
      expect(payload.client_tools.map((tool) => tool.name)).toEqual([
        CANVAS_UI_CLIENT_TOOL_NAME,
        LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
        REQUEST_SECRET_TOOL_NAME,
      ]);
    });

    it("drops conflicting user-supplied Canvas module qualnames", () => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: {
          ...DEFAULT_SETTINGS,
          conversation_settings: {
            ...DEFAULT_SETTINGS.conversation_settings,
            tool_module_qualnames: {
              canvas_ui: "custom_canvas_ui_tool",
              [CANVAS_UI_CLIENT_TOOL_NAME]: "custom_canvas_ui_control_tool",
              my_tool: "my_package.my_tool",
            },
          },
        },
      }) as { tool_module_qualnames: Record<string, string> };

      expect(payload.tool_module_qualnames).toEqual({
        my_tool: "my_package.my_tool",
      });
    });

    it("preserves non-Canvas custom tool modules", () => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: {
          ...DEFAULT_SETTINGS,
          conversation_settings: {
            ...DEFAULT_SETTINGS.conversation_settings,
            tool_module_qualnames: { my_tool: "my_package.my_tool" },
          },
        },
      }) as { tool_module_qualnames: Record<string, string> };

      expect(payload.tool_module_qualnames).toEqual({
        my_tool: "my_package.my_tool",
      });
    });
  });
});

describe("getDefaultConversationTitle", () => {
  it("formats the title using the first 5 characters of the conversation id", () => {
    expect(getDefaultConversationTitle("372eb-1234-5678-9abc")).toBe(
      "Conversation 372eb",
    );
  });
});

describe("buildWorkspaceTags (#17386)", () => {
  it("emits the workspace tag with a normalized path", () => {
    expect(buildWorkspaceTags("/Users/jane/projects/openhands/")).toEqual({
      [WORKSPACE_TAG_KEY]: "/Users/jane/projects/openhands",
    });
  });

  it("returns an empty object for null/empty/whitespace input", () => {
    expect(buildWorkspaceTags(null)).toEqual({});
    expect(buildWorkspaceTags(undefined)).toEqual({});
    expect(buildWorkspaceTags("   ")).toEqual({});
  });

  it("only ever emits keys the agent-server accepts (`^[a-z0-9]+$`)", () => {
    const tags = buildWorkspaceTags("/Users/jane/projects/openhands");
    for (const key of Object.keys(tags)) {
      expect(key).toMatch(/^[a-z0-9]+$/);
    }
  });
});

describe("toAppConversation", () => {
  const baseInfo: DirectConversationInfo = {
    id: "372eb-1234-5678-9abc",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };

  describe("server-side workspace tag fallback (#17386)", () => {
    afterEach(() => {
      removeStoredConversationMetadata(baseInfo.id);
    });

    it("uses a key the agent-server will accept (lowercase alphanumeric)", () => {
      // The agent-server rejects tag keys that are not `^[a-z0-9]+$` (SDK
      // conversation/types.py), which would 422 the create request.
      expect(WORKSPACE_TAG_KEY).toMatch(/^[a-z0-9]+$/);
    });

    it("hydrates selected_workspace from the tag when localStorage has none", () => {
      const result = toAppConversation({
        ...baseInfo,
        tags: { [WORKSPACE_TAG_KEY]: "/Users/jane/projects/openhands" },
      });

      expect(result.selected_workspace).toBe("/Users/jane/projects/openhands");
    });

    it("prefers stored metadata over the tag (warm cache for the creating client)", () => {
      setStoredConversationMetadata(baseInfo.id, {
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        selected_workspace: "/Users/jane/projects/local-fork",
      });

      const result = toAppConversation({
        ...baseInfo,
        tags: { [WORKSPACE_TAG_KEY]: "/Users/jane/projects/openhands" },
      });

      expect(result.selected_workspace).toBe("/Users/jane/projects/local-fork");
    });

    it("stays null when neither localStorage nor the tag has a workspace", () => {
      const result = toAppConversation({ ...baseInfo, tags: {} });

      expect(result.selected_workspace).toBeNull();
    });

    it("hydrates repo, branch and provider from the tags (#15598)", () => {
      const result = toAppConversation({
        ...baseInfo,
        tags: {
          repository: "OpenHands/OpenHands",
          selectedbranch: "main",
          gitprovider: "github",
        },
      });

      expect(result.selected_repository).toBe("OpenHands/OpenHands");
      expect(result.selected_branch).toBe("main");
      expect(result.git_provider).toBe("github");
    });

    it("drops an unrecognized gitprovider tag instead of casting it", () => {
      const result = toAppConversation({
        ...baseInfo,
        tags: { gitprovider: "not-a-real-provider" },
      });

      expect(result.git_provider).toBeNull();
    });
  });

  it("uses the archived status for a missing runtime that cannot resume", () => {
    const result = toAppConversation({
      ...baseInfo,
      runtime_info: {
        runtime_status: "missing",
        can_resume: false,
      },
    });

    expect(result.sandbox_status).toBe("MISSING");
  });

  it("does not archive a missing runtime that can resume", () => {
    const result = toAppConversation({
      ...baseInfo,
      runtime_info: {
        runtime_status: "missing",
        can_resume: true,
      },
    });

    expect(result.sandbox_status).toBeNull();
  });

  it("combines stats.usage_to_metrics into metrics when the backend doesn't set metrics directly (#16480)", () => {
    const result = toAppConversation({
      ...baseInfo,
      stats: {
        usage_to_metrics: {
          agent: {
            model_name: "agent-model",
            accumulated_cost: 1.5,
            max_budget_per_task: 10,
            accumulated_token_usage: {
              prompt_tokens: 100,
              completion_tokens: 20,
              cache_read_tokens: 5,
              cache_write_tokens: 1,
              context_window: 8000,
              per_turn_token: 120,
            },
            costs: [],
            response_latencies: [],
            token_usages: [],
          },
          condenser: {
            model_name: "condenser-model",
            accumulated_cost: 0.5,
            max_budget_per_task: null,
            accumulated_token_usage: {
              prompt_tokens: 40,
              completion_tokens: 10,
              cache_read_tokens: 0,
              cache_write_tokens: 0,
              context_window: 4000,
              per_turn_token: 50,
            },
            costs: [],
            response_latencies: [],
            token_usages: [],
          },
        },
      },
    });

    expect(result.metrics).toEqual({
      accumulated_cost: 2,
      max_budget_per_task: 10,
      accumulated_token_usage: {
        prompt_tokens: 140,
        completion_tokens: 30,
        cache_read_tokens: 5,
        cache_write_tokens: 1,
        context_window: 8000,
        per_turn_token: 120,
      },
    });
  });

  it("prefers backend-provided metrics over stats.usage_to_metrics when both are present", () => {
    const result = toAppConversation({
      ...baseInfo,
      metrics: { accumulated_cost: 3, max_budget_per_task: null },
      stats: {
        usage_to_metrics: {
          agent: {
            model_name: "agent-model",
            accumulated_cost: 999,
            max_budget_per_task: null,
            accumulated_token_usage: null,
            costs: [],
            response_latencies: [],
            token_usages: [],
          },
        },
      },
    });

    expect(result.metrics?.accumulated_cost).toBe(3);
  });

  it("defaults metrics to a zero-cost snapshot when neither metrics nor stats are present", () => {
    const result = toAppConversation({ ...baseInfo });

    expect(result.metrics).toEqual({
      accumulated_cost: 0,
      max_budget_per_task: null,
      accumulated_token_usage: null,
    });
  });

  it("falls back to the default title when the backend returns null", () => {
    const result = toAppConversation({ ...baseInfo, title: null });
    expect(result.title).toBe("Conversation 372eb");
  });

  it("falls back to the default title when the backend returns undefined", () => {
    const result = toAppConversation({ ...baseInfo });
    expect(result.title).toBe("Conversation 372eb");
  });

  it("falls back to the default title when the backend returns an empty string", () => {
    const result = toAppConversation({ ...baseInfo, title: "" });
    expect(result.title).toBe("Conversation 372eb");
  });

  it("falls back to the default title when the backend returns whitespace only", () => {
    const result = toAppConversation({ ...baseInfo, title: "   " });
    expect(result.title).toBe("Conversation 372eb");
  });

  it("preserves a backend-provided title when one is set", () => {
    const result = toAppConversation({
      ...baseInfo,
      title: "My real title",
    });
    expect(result.title).toBe("My real title");
  });

  it("hydrates selected_workspace from stored metadata so the sidebar can group by it", () => {
    setStoredConversationMetadata(baseInfo.id, {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      selected_workspace: "/workspace/agent-server-gui",
    });
    try {
      const result = toAppConversation({
        ...baseInfo,
        workspace: { working_dir: "/workspace/agent-server-gui/wt-abc" },
      });
      expect(result.selected_workspace).toBe("/workspace/agent-server-gui");
    } finally {
      removeStoredConversationMetadata(baseInfo.id);
    }
  });

  it("hydrates active_profile from stored metadata so the switcher shows the exact profile (#1082)", () => {
    setStoredConversationMetadata(baseInfo.id, {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      active_profile: "claude-sonnet-4.6",
    });
    try {
      const result = toAppConversation({
        ...baseInfo,
        agent: {
          kind: "Agent",
          llm: { model: "openhands/claude-sonnet-4-6" },
        },
      });
      expect(result.active_profile).toBe("claude-sonnet-4.6");
    } finally {
      removeStoredConversationMetadata(baseInfo.id);
    }
  });

  it("marks openhands conversations and surfaces the agent.llm.model", () => {
    const result = toAppConversation({
      ...baseInfo,
      agent: { kind: "Agent", llm: { model: "claude-sonnet-4-6" } },
    });
    expect(result.agent_kind).toBe("openhands");
    expect(result.llm_model).toBe("claude-sonnet-4-6");
  });

  it("reports no model when an openhands conversation has no agent LLM", () => {
    // Naming DEFAULT_SETTINGS.llm_model here invented a model the session was
    // not running; it matched no saved profile, so the chat-input picker
    // collapsed to its "Select a model" placeholder (#16263).
    const result = toAppConversation(baseInfo);

    expect(result.agent_kind).toBe("openhands");
    expect(result.llm_model).toBeNull();
  });

  it("marks ACP conversations and surfaces the configured acp_model", () => {
    // The SDK's ACPAgent may still carry a sentinel ``llm`` (``acp-managed``)
    // for cost-attribution. Consumers should see the concrete ACP model Canvas
    // configured, while SwitchProfileButton remains gated by agent_kind.
    const result = toAppConversation({
      ...baseInfo,
      agent: {
        kind: "ACPAgent",
        acp_model: "claude-sonnet-4-6",
        llm: { model: "acp-managed" },
      },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.llm_model).toBe("claude-sonnet-4-6");
  });

  it("prefers ACP runtime model fields over configured acp_model", () => {
    const result = toAppConversation({
      ...baseInfo,
      current_model_id: "claude-sonnet-4-6",
      current_model_name: "Claude Sonnet",
      agent: {
        kind: "ACPAgent",
        acp_model: "claude-opus-4-7",
        llm: { model: "acp-managed" },
      },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.llm_model).toBe("Claude Sonnet");
  });

  it("surfaces the runtime ACP default model over a configured acp_model", () => {
    // claude-agent-acp 0.44+ exposes ``default`` ("Default (recommended)")
    // as a real, selectable model in its configOptions select. The runtime
    // ``current_model_*`` fields take precedence over the configured
    // ``acp_model``, so a session actually running on ``default`` must
    // surface that on the chip instead of the stale configured value.
    const result = toAppConversation({
      ...baseInfo,
      current_model_id: "default",
      current_model_name: "Default (recommended)",
      agent: {
        kind: "ACPAgent",
        acp_model: "claude-sonnet-4-6",
        llm: { model: "acp-managed" },
      },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.llm_model).toBe("Default (recommended)");
  });

  it("falls back to a non-sentinel ACP llm.model for SDKs that mirror acp_model there", () => {
    const result = toAppConversation({
      ...baseInfo,
      agent: { kind: "ACPAgent", llm: { model: "claude-sonnet-4-6" } },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.llm_model).toBe("claude-sonnet-4-6");
  });

  it("surfaces ACP default model surfaced via the configured acp_model", () => {
    // ``default`` / ``Default (recommended)`` is a real claude-agent-acp
    // model id (not an SDK placeholder), so it must surface like any other
    // configured model.
    const result = toAppConversation({
      ...baseInfo,
      agent: {
        kind: "ACPAgent",
        acp_model: "Default (recommended)",
        llm: { model: "acp-managed" },
      },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.llm_model).toBe("Default (recommended)");
  });

  it("surfaces ACP default model surfaced via agent.llm.model", () => {
    // Same as above, one rung lower in the precedence chain: ``default`` is
    // a real model id and should surface as-is.
    const result = toAppConversation({
      ...baseInfo,
      agent: { kind: "ACPAgent", llm: { model: "default" } },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.llm_model).toBe("default");
  });

  it("surfaces acp_server from tags.acpserver for ACP conversations", () => {
    // The ``acpserver`` conversation tag is stamped at create time
    // (``buildStartConversationRequest``) but never previously plumbed
    // through on read — the sidebar chip in agent-canvas#405 needs this
    // value to resolve the human display name ("Claude Code" / "Codex" /
    // "Gemini CLI").
    const result = toAppConversation({
      ...baseInfo,
      agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
      tags: { [ACP_SERVER_TAG_KEY]: "claude-code" },
    });
    expect(result.acp_server).toBe("claude-code");
  });

  it("leaves acp_server null when an ACP conversation has no tag stamped", () => {
    // Older conversations created before the tag was added, or ACP
    // conversations created via the raw API, won't have the tag. The
    // sidebar should still render a chip ("ACP") — but the resolver gets
    // null here and the UI fallback handles the generic label.
    const result = toAppConversation({
      ...baseInfo,
      agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.acp_server).toBeNull();
  });

  it("ignores tags.acpserver on OpenHands conversations to prevent stray-tag bleed", () => {
    // The agent-server's pydantic model doesn't enforce that ``acpserver``
    // is only stamped on ACP conversations. Defensively gating on
    // ``agent.kind === "ACPAgent"`` keeps a misconfigured tag from
    // turning the sidebar of an OpenHands conversation into "Claude
    // Code". Pairs with the ``llm_model`` null-out for ACP.
    const result = toAppConversation({
      ...baseInfo,
      agent: { kind: "Agent", llm: { model: "claude-sonnet-4-6" } },
      tags: { [ACP_SERVER_TAG_KEY]: "claude-code" },
    });
    expect(result.agent_kind).toBe("openhands");
    expect(result.acp_server).toBeNull();
  });
});

describe("buildRuntimeServicesSystemSuffix", () => {
  it("returns undefined when runtime services info is absent", () => {
    expect(buildRuntimeServicesSystemSuffix()).toBeUndefined();
  });

  it("parses runtime services JSON strings", () => {
    expect(
      parseRuntimeServicesInfo(
        JSON.stringify({
          mode: "dev:automation",
          services: {
            agent_server: { url_from_agent: "http://localhost:18000" },
          },
        }),
      )?.mode,
    ).toBe("dev:automation");
  });

  it("returns null when runtime services JSON is malformed", () => {
    expect(parseRuntimeServicesInfo("{not valid json")).toBeNull();
  });

  it("returns undefined when the JSON has no services", () => {
    expect(buildRuntimeServicesSystemSuffix({ mode: "x" })).toBeUndefined();
  });

  it("renders a <RUNTIME_SERVICES> block when an automation entry is present", () => {
    const suffix = buildRuntimeServicesSystemSuffix({
      mode: "dev:automation",
      agent_host_alias: "localhost",
      services: {
        agent_server: {
          description: "self",
          url_from_agent: "http://localhost:18000",
        },
        automation: {
          description: "automations",
          url_from_agent: "http://localhost:18001",
          api_prefix: "/api/automation",
          docs_url: "http://localhost:18001/api/automation/docs",
          openapi_url: "http://localhost:18001/api/automation/openapi.json",
          auth_env_var: "OPENHANDS_AUTOMATION_API_KEY",
        },
      },
    });
    expect(suffix).toBeDefined();
    expect(suffix).toContain("<RUNTIME_SERVICES>");
    expect(suffix).toContain("dev:automation");
    expect(suffix).toContain("http://localhost:18000");
    expect(suffix).toContain("http://localhost:18001");
    expect(suffix).toContain("http://localhost:18001/api/automation/docs");
    expect(suffix).toContain(
      "X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY",
    );
    expect(suffix).not.toContain("X-API-Key: $OPENHANDS_AUTOMATION_API_KEY");
    expect(suffix).toContain("</RUNTIME_SERVICES>");
    // The "don't guess" line should reference the actual agent-server URL
    // for this stack, not a hardcoded port. The assertion anchors on the URL
    // we supplied above.
    expect(suffix).toContain(
      "In particular, http://localhost:18000 inside your sandbox is the Agent Server",
    );
  });

  it("uses the configured agent-server URL in the don't-guess line (not a hardcoded :8000)", () => {
    // dev:safe runs the agent-server on :18000, not :8000. Make sure the
    // rendered block doesn't lie to the agent about its own URL.
    const suffix = buildRuntimeServicesSystemSuffix({
      mode: "dev:safe",
      services: {
        agent_server: { url_from_agent: "http://localhost:18000" },
      },
    });
    expect(suffix).toBeDefined();
    expect(suffix).toContain(
      "In particular, http://localhost:18000 inside your sandbox is the Agent Server",
    );
    expect(suffix).not.toContain(
      "In particular, http://localhost:8000 inside your sandbox",
    );
  });

  it("renders the frontend entry with the new key", () => {
    const suffix = buildRuntimeServicesSystemSuffix({
      mode: "dev:static",
      services: {
        agent_server: { url_from_agent: "http://localhost:18000" },
        frontend: {
          kind: "static",
          description: "Static-file server hosting the agent-canvas build.",
          url_from_agent: "http://localhost:3001",
        },
      },
    });
    expect(suffix).toContain("* Frontend: http://localhost:3001");
    expect(suffix).toContain("Static-file server");
    // Should NOT mislabel a static-build frontend as "Vite frontend".
    expect(suffix).not.toContain("Vite frontend");
  });

  it("explicitly mentions when automation is absent", () => {
    const suffix = buildRuntimeServicesSystemSuffix({
      mode: "dev:safe",
      services: {
        agent_server: { url_from_agent: "http://localhost:18000" },
      },
    });
    expect(suffix).toBeDefined();
    expect(suffix).toContain("Automation backend: not running");
  });

  it("reads server_info from the cached probe when available", async () => {
    mockGetCachedAgentServerInfo.mockReturnValue({
      version: "1.48.0",
      execution_runtime: "docker",
    });

    const runtime = await fetchBackendExecutionRuntime();

    expect(runtime).toBe("docker");
    expect(mockGetServerInfo).not.toHaveBeenCalled();
  });

  it("fetches /server_info when there is no cached probe", async () => {
    mockGetServerInfo.mockResolvedValue({
      version: "1.48.0",
      execution_runtime: "local",
    });

    const runtime = await fetchBackendExecutionRuntime();

    expect(runtime).toBe("local");
    expect(mockGetServerInfo).toHaveBeenCalledOnce();
  });
});

type LaunchAdditions = {
  system_message_suffix_append?: string;
  llm_profile_ref?: string;
  skills?: Array<{ name: string; source: string; content: string }>;
};

const additionsOf = (payload: unknown): LaunchAdditions | undefined =>
  (payload as { agent_launch_additions?: LaunchAdditions })
    .agent_launch_additions;

const RUNTIME_SERVICES = {
  mode: "dev:automation",
  services: {
    agent_server: { url_from_agent: "http://localhost:18000" },
    automation: { url_from_agent: "http://localhost:18001" },
  },
};

describe("profile launch additions", () => {
  it("carries runtime services and leaves the saved global suffix to the profile", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          agent_context: { system_message_suffix: "MARKER-GLOBAL-RULES" },
        },
      },
      query: "hello",
      runtimeServicesInfo: RUNTIME_SERVICES,
    });

    const appended = additionsOf(payload)?.system_message_suffix_append;
    expect(payload.agent_profile_id).toBe("profile-1");
    expect(payload).not.toHaveProperty("agent_settings");
    expect(appended).toMatch(/^<RUNTIME_SERVICES>/);
    expect(appended).toContain("http://localhost:18001");
    expect(appended).not.toContain("MARKER-GLOBAL-RULES");
  });

  it("carries runtime services on an ACP profile launch too", () => {
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
      query: "hello",
      agentProfileId: "profile-acp",
      agentProfileKind: "acp",
      runtimeServicesInfo: RUNTIME_SERVICES,
    });

    expect(payload.agent_profile_id).toBe("profile-acp");
    expect(additionsOf(payload)?.system_message_suffix_append).toContain(
      "<RUNTIME_SERVICES>",
    );
  });

  it.each([
    ["the toggle is on and the profile carries a router", true, true, true],
    ["the profile carries no router", true, false, false],
    ["the toggle is off", false, true, false],
  ])(
    "routes the first message only when %s",
    (_case, runAtStart, profileHasRouter, expected) => {
      const payload = buildStartConversationRequest({
        ...PROFILE,
        settings: {
          ...DEFAULT_SETTINGS,
          run_router_at_conversation_start: runAtStart,
        },
        profileHasRouter,
        query: "hello",
      });

      const appended = additionsOf(payload)?.system_message_suffix_append;
      if (expected) {
        expect(appended).toContain("<ROUTE_AT_CONVERSATION_START>");
      } else {
        expect(appended ?? "").not.toContain("ROUTE_AT_CONVERSATION_START");
      }
    },
  );

  it("suppresses the route-at-start suffix when the home launch passes suppressRouterAtStart", () => {
    // The flag is threaded in by useCreateConversation (read-and-cleared from
    // the store) rather than read by the builder, so child/`/new` launches
    // that don't pass it route as usual.
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        run_router_at_conversation_start: true,
      },
      profileHasRouter: true,
      suppressRouterAtStart: true,
      query: "hello",
    });
    const appended = additionsOf(payload)?.system_message_suffix_append ?? "";
    expect(appended).not.toContain("ROUTE_AT_CONVERSATION_START");
  });

  it("does not suppress route-at-start when suppressRouterAtStart is omitted", () => {
    // A child or `/new` launch omits the flag — routing is unaffected by any
    // pending home pick.
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        run_router_at_conversation_start: true,
      },
      profileHasRouter: true,
      query: "hello",
    });
    const appended = additionsOf(payload)?.system_message_suffix_append ?? "";
    expect(appended).toContain("<ROUTE_AT_CONVERSATION_START>");
  });

  it("adds a catalog skill the opening message invokes", () => {
    const [entry] = SKILLS_CATALOG;
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      query: `/${entry.name} please`,
    });

    const skills = additionsOf(payload)?.skills ?? [];
    expect(skills.map((skill) => skill.name)).toEqual([entry.name]);
    expect(skills[0].content).toBe(entry.content);
    expect(skills[0].source).toMatch(new RegExp(`/${entry.name}/SKILL\\.md$`));
  });

  it("adds the invoked catalog skill to an ACP launch", () => {
    const [entry] = SKILLS_CATALOG;
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
      agentProfileId: "profile-acp",
      agentProfileKind: "acp",
      query: `/${entry.name} please`,
    });

    expect(additionsOf(payload)?.skills?.map((skill) => skill.name)).toContain(
      entry.name,
    );
  });

  it("adds no skills for a message that invokes no catalog skill", () => {
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
      agentProfileId: "profile-1",
      agentProfileKind: "openhands",
      query: "/not-a-catalog-skill please",
    });

    expect(additionsOf(payload)?.skills).toBeUndefined();
  });

  it("omits additions when there is nothing to add", () => {
    const payload = buildStartConversationRequest({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      query: "hello",
    });

    expect(additionsOf(payload)).toBeUndefined();
  });
});

describe("buildStartConversationRequestWithEncryptedSettings", () => {
  let settingsSpy: MockInstance;
  let secretsSpy: MockInstance;

  beforeEach(() => {
    mockLoadHooks.mockReset();
    // Stub only the collaborators; the hooks lookup is what's under test.
    settingsSpy = vi
      .spyOn(SettingsService, "getSettingsForConversation")
      .mockResolvedValue({
        agentSettings: (DEFAULT_SETTINGS.agent_settings ?? {}) as Record<
          string,
          SettingsValue
        >,
        conversationSettings: (DEFAULT_SETTINGS.conversation_settings ??
          {}) as Record<string, SettingsValue>,
        secretsEncrypted: true,
        skillEnablement: { disabledSkills: [] },
      });
    secretsSpy = vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
  });

  // vitest is not configured with `restoreMocks`, so restore explicitly.
  afterEach(() => {
    settingsSpy.mockRestore();
    secretsSpy.mockRestore();
  });

  it("looks hooks up in hooksProjectDir, not the conversation working dir", async () => {
    mockLoadHooks.mockResolvedValue({ hook_config: WORKSPACE_HOOK_CONFIG });

    const payload = await buildStartConversationRequestWithEncryptedSettings({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      workingDir: "/workspace/my-project/0f1e2d3c",
      hooksProjectDir: "/workspace/my-project",
    });

    expect(mockLoadHooks).toHaveBeenCalledWith({
      project_dir: "/workspace/my-project",
    });
    expect(payload.hook_config).toEqual(WORKSPACE_HOOK_CONFIG);
  });

  it("falls back to the configured working dir when no hooksProjectDir is given", async () => {
    mockLoadHooks.mockResolvedValue({ hook_config: null });

    await buildStartConversationRequestWithEncryptedSettings({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      workingDir: "/workspace/my-project/0f1e2d3c",
    });

    expect(mockLoadHooks).toHaveBeenCalledWith({
      project_dir: "/workspace/project/agent-canvas",
    });
  });

  it("omits hook_config when the workspace has no hooks", async () => {
    mockLoadHooks.mockResolvedValue({ hook_config: null });

    const payload = await buildStartConversationRequestWithEncryptedSettings({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      hooksProjectDir: "/workspace/my-project",
    });

    expect(payload.hook_config).toBeUndefined();
  });

  it("starts the conversation without hooks when the lookup fails", async () => {
    mockLoadHooks.mockRejectedValue(new Error("500 Internal Server Error"));

    const payload = await buildStartConversationRequestWithEncryptedSettings({
      ...PROFILE,
      settings: DEFAULT_SETTINGS,
      hooksProjectDir: "/workspace/my-project",
    });

    expect(payload.hook_config).toBeUndefined();
    expect(payload.agent_profile_id).toBe("profile-1");
  });

  describe("skills for an ACP launch", () => {
    const launch = () =>
      buildStartConversationRequestWithEncryptedSettings({
        settings: { ...DEFAULT_SETTINGS, enabled_skills: [first.name] },
        agentProfileId: "profile-acp",
        agentProfileKind: "acp",
        hooksProjectDir: null,
      });
    const skillNames = (payload: unknown) =>
      (additionsOf(payload)?.skills ?? []).map((skill) => skill.name);
    const [first, second] = SKILLS_CATALOG;

    afterEach(() => {
      vi.mocked(AgentProfilesService.getProfile).mockRestore();
    });

    it("follows the OpenHands default profile that owns the Skills page", async () => {
      vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
        profile: {
          agent_kind: "openhands",
          public_skills: [first.name, second.name],
          disabled_skills: [first.name],
        },
      } as unknown as Awaited<
        ReturnType<typeof AgentProfilesService.getProfile>
      >);

      expect(skillNames(await launch())).toEqual([second.name]);
    });

    it("falls back to the settings lists when there is no default profile", async () => {
      vi.spyOn(AgentProfilesService, "getProfile").mockRejectedValue(
        new HttpError(404, "Not Found"),
      );

      expect(skillNames(await launch())).toEqual([first.name]);
    });

    it("uses the settings selection when the default profile is ACP", async () => {
      vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
        profile: { agent_kind: "acp" },
      } as Awaited<ReturnType<typeof AgentProfilesService.getProfile>>);

      expect(skillNames(await launch())).toEqual([first.name]);
    });

    it.each([new HttpError(500, "Server error"), new Error("Network error")])(
      "rejects the launch when the authoritative selection cannot be read: %s",
      async (error) => {
        vi.spyOn(AgentProfilesService, "getProfile").mockRejectedValue(error);

        await expect(launch()).rejects.toBe(error);
      },
    );
  });

  it("routes the first message for a profile that carries a router", async () => {
    const payload = await buildStartConversationRequestWithEncryptedSettings({
      ...PROFILE,
      settings: {
        ...DEFAULT_SETTINGS,
        run_router_at_conversation_start: true,
      },
      profileHasRouter: true,
      hooksProjectDir: "/workspace/my-project",
    });

    expect(additionsOf(payload)?.system_message_suffix_append).toContain(
      "ROUTE_AT_CONVERSATION_START",
    );
  });
});

describe("toHooksResponse", () => {
  // Prompt and agent hooks reach Canvas over the wire with `prompt`/`system_prompt`,
  // which the typed client's `HookDefinition` does not declare yet.
  const wireHook = (hook: Record<string, unknown>) =>
    hook as unknown as HookConfig["stop"][number]["hooks"][number];

  it("carries a prompt hook's prompt through", () => {
    const response = toHooksResponse(
      makeHookConfig({
        stop: [
          {
            matcher: "*",
            hooks: [
              wireHook({
                type: "prompt",
                command: "",
                prompt: "QA_F07 prompt hook",
                system_prompt: null,
                timeout: 60,
                async: false,
              }),
            ],
          },
        ],
      }),
    );

    expect(response.hooks).toEqual([
      {
        event_type: "stop",
        matchers: [
          {
            matcher: "*",
            hooks: [
              {
                type: "prompt",
                command: "",
                prompt: "QA_F07 prompt hook",
                timeout: 60,
                async: false,
              },
            ],
          },
        ],
      },
    ]);
  });

  it("carries an agent hook's system_prompt through", () => {
    const response = toHooksResponse(
      makeHookConfig({
        session_start: [
          {
            matcher: "*",
            hooks: [
              wireHook({
                type: "agent",
                command: "",
                prompt: null,
                system_prompt: "QA_F07 agent hook",
                timeout: 60,
                async: false,
              }),
            ],
          },
        ],
      }),
    );

    expect(response.hooks).toEqual([
      {
        event_type: "session_start",
        matchers: [
          {
            matcher: "*",
            hooks: [
              {
                type: "agent",
                command: "",
                system_prompt: "QA_F07 agent hook",
                timeout: 60,
                async: false,
              },
            ],
          },
        ],
      },
    ]);
  });

  it("keeps command hooks unchanged", () => {
    const response = toHooksResponse(
      makeHookConfig({
        pre_tool_use: [
          {
            matcher: "terminal",
            hooks: [{ type: HookType.COMMAND, command: "true", timeout: 10 }],
          },
        ],
      }),
    );

    expect(response.hooks).toEqual([
      {
        event_type: "pre_tool_use",
        matchers: [
          {
            matcher: "terminal",
            hooks: [{ type: HookType.COMMAND, command: "true", timeout: 10 }],
          },
        ],
      },
    ]);
  });
});
