import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CANVAS_UI_CLIENT_TOOL_NAME } from "#/constants/canvas-ui";

import {
  ACP_SERVER_TAG_KEY,
  buildRuntimeServicesSystemSuffix,
  buildStartConversationRequest,
  getDefaultConversationTitle,
  toAppConversation,
  type DirectConversationInfo,
} from "#/api/agent-server-adapter";
import {
  removeStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { ACP_VERTEX_SAFE_MODEL } from "#/constants/acp-providers";
import { DEFAULT_SETTINGS } from "#/services/settings";
import {
  LLM_AUTH_TYPE_SUBSCRIPTION,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";

const {
  mockGetAgentServerWorkingDir,
  mockIsAgentServerToolAvailable,
  mockGetEffectiveLocalBackend,
} = vi.hoisted(() => ({
  mockGetAgentServerWorkingDir: vi.fn(() => "/workspace/project/agent-canvas"),
  mockIsAgentServerToolAvailable: vi.fn((_toolName: string) => true),
  mockGetEffectiveLocalBackend: vi.fn(() => ({
    id: "default-local",
    name: "Local backend",
    host: "http://127.0.0.1:8000",
    apiKey: "session-key",
    kind: "local" as const,
  })),
}));

vi.mock("#/api/agent-server-config", () => ({
  getAgentServerBaseUrl: vi.fn(() => "http://127.0.0.1:8000"),
  getAgentServerSessionApiKey: vi.fn(() => null),
  getAgentServerWorkingDir: mockGetAgentServerWorkingDir,
  shouldLoadPublicSkills: vi.fn(() => true),
  syncBakedSessionApiKey: vi.fn(),
}));

vi.mock("#/api/agent-server-compatibility", () => ({
  isAgentServerToolAvailable: mockIsAgentServerToolAvailable,
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getEffectiveLocalBackend: mockGetEffectiveLocalBackend,
}));

beforeEach(() => {
  mockIsAgentServerToolAvailable.mockReturnValue(true);
  mockGetEffectiveLocalBackend.mockReturnValue({
    id: "default-local",
    name: "Local backend",
    host: "http://127.0.0.1:8000",
    apiKey: "session-key",
    kind: "local",
  });
});

describe("buildStartConversationRequest", () => {
  it("includes the resolved title LLM profile when configured", () => {
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
      titleLlmProfile: "Titles",
    });

    expect(payload.title_llm_profile).toBe("Titles");
  });

  it("omits the title LLM profile in automatic mode without a default", () => {
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
    });

    expect(payload).not.toHaveProperty("title_llm_profile");
  });

  it("uses nested settings as the source of truth and lets the SDK create the agent", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        llm_model: "stale-top-level-model",
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          agent: "CodeActAgent",
          enable_sub_agents: true,
          llm: {
            model: "nested-model",
            api_key: "  nested-key  ",
            base_url: " https://nested.example.com ",
          },
          condenser: {
            enabled: true,
            max_size: 120,
          },
          enable_switch_llm_tool: true,
        },
        conversation_settings: {
          ...DEFAULT_SETTINGS.conversation_settings,
          max_iterations: 123,
        },
      },
      query: "hello",
    }) as {
      agent?: unknown;
      agent_settings: Record<string, unknown> & {
        llm: Record<string, unknown>;
        tools: Array<{ name: string; params: Record<string, unknown> }>;
        agent_context: Record<string, unknown>;
      };
      workspace: { working_dir: string };
      initial_message: { content: Array<{ text: string }> };
      max_iterations: number;
    };

    expect(payload.agent).toBeUndefined();
    expect(payload.agent_settings.llm).toMatchObject({
      model: "nested-model",
      api_key: "nested-key",
      base_url: "https://nested.example.com",
      // Streaming is enabled so the OpenHands agent emits StreamingDeltaEvents.
      stream: true,
    });
    expect(payload.agent_settings.condenser).toEqual({
      enabled: true,
      max_size: 120,
    });
    expect(payload.agent_settings.tools).toEqual([
      { name: "terminal", params: {} },
      { name: "file_editor", params: {} },
      { name: "task_tracker", params: {} },
      { name: "browser_tool_set", params: {} },
      { name: "task_tool_set", params: {} },
    ]);
    expect(payload.agent_settings.agent_context).toMatchObject({
      load_public_skills: false,
      load_user_skills: true,
      load_project_skills: true,
    });
    // Bundled public skills are injected into agent_context.skills so the
    // SDK can perform trigger matching without cloning the extensions repo.
    expect(Array.isArray(payload.agent_settings.agent_context.skills)).toBe(
      true,
    );
    const skills = payload.agent_settings.agent_context.skills as Record<
      string,
      unknown
    >[];
    expect(skills.length).toBeGreaterThan(0);
    // Every bundled skill must carry the fields the SDK needs for trigger
    // matching and system-prompt injection.
    for (const skill of skills) {
      expect(skill).toHaveProperty("name");
      expect(skill).toHaveProperty("content");
      // source must be an absolute path to the skill's SKILL.md so the
      // Python agent-server can resolve bundled resources (scripts/, references/).
      const source = skill.source as string;
      expect(source).toMatch(/^\//);
      expect(source).toMatch(
        new RegExp(`/${skill.name as string}/SKILL\\.md$`),
      );
      expect(skill).toHaveProperty("is_agentskills_format", true);
      // trigger is either null (always-active) or { type, keywords }
      if (skill.trigger !== null) {
        expect(skill.trigger).toMatchObject({
          type: "keyword",
          keywords: expect.arrayContaining([expect.any(String)]),
        });
      }
    }
    expect(payload.agent_settings.agent).toBe("CodeActAgent");
    expect(payload.agent_settings.enable_switch_llm_tool).toBe(true);
    expect(payload.workspace.working_dir).toBe(
      "/workspace/project/agent-canvas",
    );
    expect(payload.max_iterations).toBe(123);
    expect(payload.initial_message.content[0]?.text).toBe("hello");
  });

  it("uses subscription auth metadata without API credentials", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: {
            model: "gpt-5.2-codex",
            api_key: "stale-api-key",
            base_url: "https://api.openai.com/v1",
            auth_type: LLM_AUTH_TYPE_SUBSCRIPTION,
            subscription_vendor: OPENAI_SUBSCRIPTION_VENDOR,
          },
        },
      },
    }) as { agent_settings: { llm: Record<string, unknown> } };

    expect(payload.agent_settings.llm).toEqual({
      model: "gpt-5.2-codex",
      stream: true,
      auth_type: LLM_AUTH_TYPE_SUBSCRIPTION,
      subscription_vendor: OPENAI_SUBSCRIPTION_VENDOR,
    });
  });

  it("passes the stored model through unchanged for subscription auth", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: {
            model: "openai/gpt-4o",
            auth_type: LLM_AUTH_TYPE_SUBSCRIPTION,
            subscription_vendor: OPENAI_SUBSCRIPTION_VENDOR,
          },
        },
      },
    }) as { agent_settings: { llm: Record<string, unknown> } };

    expect(payload.agent_settings.llm.model).toBe("openai/gpt-4o");
  });

  it("forwards the switch-LLM setting to SDK agent settings", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          enable_switch_llm_tool: true,
          llm: { model: "nested-model" },
        },
      },
    }) as {
      agent?: unknown;
      agent_settings: {
        enable_switch_llm_tool?: boolean;
        include_default_tools?: unknown;
      };
    };

    expect(payload.agent).toBeUndefined();
    expect(payload.agent_settings.enable_switch_llm_tool).toBe(true);
    expect(payload.agent_settings.include_default_tools).toBeUndefined();
  });

  it("omits browser_tool_set and task_tool_set when the server does not advertise them", () => {
    mockIsAgentServerToolAvailable.mockReturnValue(false);

    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          llm: { model: "nested-model" },
        },
      },
    }) as {
      agent_settings: {
        tools: Array<{ name: string; params: Record<string, unknown> }>;
      };
    };

    expect(payload.agent_settings.tools).toEqual([
      { name: "terminal", params: {} },
      { name: "file_editor", params: {} },
      { name: "task_tracker", params: {} },
    ]);
  });

  it("includes task_tool_set when sub-agents are enabled and the server advertises it but not browser tools", () => {
    mockIsAgentServerToolAvailable.mockImplementation(
      (toolName: string) => toolName === "task_tool_set",
    );

    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          enable_sub_agents: true,
          llm: { model: "nested-model" },
        },
      },
    }) as {
      agent_settings: {
        tools: Array<{ name: string; params: Record<string, unknown> }>;
      };
    };

    expect(payload.agent_settings.tools).toEqual([
      { name: "terminal", params: {} },
      { name: "file_editor", params: {} },
      { name: "task_tracker", params: {} },
      { name: "task_tool_set", params: {} },
    ]);
  });

  it("omits task_tool_set when sub-agents are disabled even if the server advertises it", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          enable_sub_agents: false,
          llm: { model: "nested-model" },
        },
      },
    }) as {
      agent_settings: {
        tools: Array<{ name: string; params: Record<string, unknown> }>;
      };
    };

    const toolNames = payload.agent_settings.tools.map((t) => t.name);
    expect(toolNames).not.toContain("task_tool_set");
  });

  it("derives confirmation and security settings the same way as OpenHands", () => {
    const payload = buildStartConversationRequest({
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

  it("serializes custom secrets as host-relative LookupSecret entries", () => {
    const payload = buildStartConversationRequest({
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

  it("does NOT mirror conversation secrets onto agent_context for ACP — request.secrets is the sole channel", () => {
    // The compatible agent-server line injects the ACP spawn env from
    // ``secret_registry``, which is seeded from ``request.secrets``
    // (sdk#3299/#3464; the agent_context drain is gone entirely in sdk#3528).
    // Mirroring the map onto ``agent_context.secrets`` would keep a second,
    // dead credential channel alive (agent-canvas#1039 wants exactly one).
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        },
      },
      customSecrets: [{ name: "ANTHROPIC_API_KEY" }],
    }) as {
      agent_settings: { agent_context?: { secrets?: Record<string, unknown> } };
      secrets: Record<string, unknown>;
    };

    expect(payload.secrets.ANTHROPIC_API_KEY).toBeDefined();
    expect(payload.agent_settings.agent_context?.secrets).toBeUndefined();
  });

  it("does not synthesize agent_context.secrets for ACP when no custom secrets are set", () => {
    // Empty/absent customSecrets must not introduce an empty
    // ``agent_context.secrets`` map on the ACPAgent payload.
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        },
      },
    }) as {
      agent_settings: { agent_context?: { secrets?: Record<string, unknown> } };
    };

    expect(payload.agent_settings.agent_context?.secrets).toBeUndefined();
  });

  it("does not mirror conversation secrets onto agent_context for non-ACP conversations", () => {
    // The OpenHands ``Agent`` reads secrets from ``secret_registry``
    // directly (no spawn-env bridging needed), so the LLM-driven path
    // must not get an extra ``agent_context.secrets`` map — that would
    // be both redundant and a surprise for any code that inspects
    // ``agent_context`` for non-secret payload (skills, suffixes, etc.).
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
      customSecrets: [{ name: "ANTHROPIC_API_KEY" }],
    }) as {
      agent_settings: { agent_context?: { secrets?: Record<string, unknown> } };
    };

    expect(payload.agent_settings.agent_context?.secrets).toBeUndefined();
  });

  describe("ACP secret delivery", () => {
    // Settings factory for a containerized ACP conversation.
    const acpSettings = (overrides: Record<string, unknown> = {}) => ({
      ...DEFAULT_SETTINGS,
      agent_settings: {
        ...DEFAULT_SETTINGS.agent_settings,
        agent_kind: "acp",
        acp_server: "codex",
        acp_command: ["npx", "-y", "@zed-industries/codex-acp"],
        acp_model: "gpt-5.5/medium",
        ...overrides,
      },
    });

    it("delivers provider credentials and user secrets uniformly as LookupSecrets in request.secrets only", () => {
      // Provider credentials (e.g. a saved CODEX_AUTH_JSON) are no longer special
      // on the wire — they ride as LookupSecrets like any custom secret. The SDK
      // resolves them off the event loop at spawn, so the loopback fetch is safe
      // (software-agent-sdk#3510). "Reserved" is now only an onboarding concept.
      const payload = buildStartConversationRequest({
        settings: acpSettings(),
        customSecrets: [{ name: "CODEX_AUTH_JSON" }, { name: "MY_TOKEN" }],
      }) as {
        agent_settings: {
          acp_model?: string;
          agent_context?: { secrets?: Record<string, { kind: string }> };
        };
        secrets: Record<string, { kind: string }>;
      };

      expect(payload.secrets.CODEX_AUTH_JSON?.kind).toBe("LookupSecret");
      expect(payload.secrets.MY_TOKEN?.kind).toBe("LookupSecret");
      // ``request.secrets`` is the sole channel — no agent_context mirror
      // (agent-server >=1.25.0 injects the spawn env from secret_registry).
      expect(payload.agent_settings.agent_context?.secrets).toBeUndefined();
      // The configured model rides along unchanged.
      expect(payload.agent_settings.acp_model).toBe("gpt-5.5/medium");
    });

    it("does NOT set secrets_encrypted for ACP — there is no encrypted payload to decrypt", () => {
      // An ACP request carries no encrypted secret (no LLM api_key; credentials
      // are LookupSecrets resolved at runtime). Flagging it encrypted hard-fails
      // ("cipher not configured") on a fresh ACP container without OH_SECRET_KEY.
      const payload = buildStartConversationRequest({
        settings: acpSettings(),
        secretsEncrypted: true,
        customSecrets: [{ name: "CODEX_AUTH_JSON" }],
      }) as { secrets_encrypted?: boolean };
      expect(payload.secrets_encrypted).toBeUndefined();
    });

    it("still sets secrets_encrypted for a non-ACP conversation (encrypted LLM key)", () => {
      const payload = buildStartConversationRequest({
        settings: DEFAULT_SETTINGS,
        secretsEncrypted: true,
      }) as { secrets_encrypted?: boolean };
      expect(payload.secrets_encrypted).toBe(true);
    });
  });

  describe("canvas_ui client tool injection", () => {
    it("sends canvas_ui as a client-defined JSON tool", () => {
      const payload = buildStartConversationRequest({
        settings: DEFAULT_SETTINGS,
      });

      expect(payload.client_tools).toHaveLength(1);
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
      expect(
        payload.agent_settings?.tools?.map((tool) => tool.name) ?? [],
      ).not.toContain("canvas_ui");
      expect(payload.tool_module_qualnames).toBeUndefined();
    });

    it("omits the client tool for an inline ACP agent", () => {
      const payload = buildStartConversationRequest({
        settings: {
          ...DEFAULT_SETTINGS,
          agent_settings: {
            ...DEFAULT_SETTINGS.agent_settings,
            agent_kind: "acp",
            acp_server: "custom",
            acp_command: ["custom-acp"],
          },
        },
      });

      expect(payload.client_tools).toEqual([]);
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
      ]);
    });

    it("sends the client tool when resuming a conversation", () => {
      const payload = buildStartConversationRequest({
        settings: DEFAULT_SETTINGS,
        conversationId: "legacy-conversation-id",
      }) as { conversation_id: string; client_tools: Array<{ name: string }> };

      expect(payload.conversation_id).toBe("legacy-conversation-id");
      expect(payload.client_tools.map((tool) => tool.name)).toEqual([
        CANVAS_UI_CLIENT_TOOL_NAME,
      ]);
    });

    it("drops conflicting user-supplied Canvas module qualnames", () => {
      const payload = buildStartConversationRequest({
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

  // @spec LLD-001 — Frontend always sends its chosen default model
  describe("llm.model fallback — frontend always sends its chosen default", () => {
    type ModelPayload = {
      agent_settings: Record<string, unknown> & {
        llm: Record<string, unknown>;
      };
    };

    function getModelFrom(
      options: Parameters<typeof buildStartConversationRequest>[0],
    ): unknown {
      return (buildStartConversationRequest(options) as unknown as ModelPayload)
        .agent_settings.llm.model;
    }

    it("uses the configured model when one is set", () => {
      expect(
        getModelFrom({
          settings: {
            ...DEFAULT_SETTINGS,
            agent_settings: {
              ...DEFAULT_SETTINGS.agent_settings,
              llm: { model: "anthropic/claude-opus-4-5" },
            },
          },
        }),
      ).toBe("anthropic/claude-opus-4-5");
    });

    // The agent-server returns '' when no model has been saved yet.
    // Without this guard the empty string passes the old typeof check and
    // the agent-server falls back to its own SDK default (gpt-5.5).
    // SettingsValue includes scalars so inline literals are type-safe here.
    it.each([
      ["undefined", { ...DEFAULT_SETTINGS.agent_settings, llm: {} }],
      [
        "an empty string",
        { ...DEFAULT_SETTINGS.agent_settings, llm: { model: "" } },
      ],
      [
        "whitespace only",
        { ...DEFAULT_SETTINGS.agent_settings, llm: { model: "   " } },
      ],
      // No llm key at all — SettingsValue accepts plain scalars.
      [
        "absent (no llm block)",
        { schema_version: 1, agent_kind: "openhands", agent: "CodeActAgent" },
      ],
      // Mirrors a fresh user who skipped onboarding: server returns {}.
      ["entirely empty", {}],
    ])(
      "falls back to DEFAULT_SETTINGS.llm_model when agent_settings.llm.model is %s",
      (_, agentSettings) => {
        expect(
          getModelFrom({
            settings: {
              ...DEFAULT_SETTINGS,
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              agent_settings: agentSettings as any,
            },
          }),
        ).toBe(DEFAULT_SETTINGS.llm_model);
      },
    );

    // encryptedAgentSettings overrides settings.agent_settings at conversation
    // start; if the encrypted payload has no model set the frontend default
    // must still be sent explicitly.
    it.each([
      ["carries an empty model", { llm: { model: "" } }],
      ["is empty", {}],
    ])(
      "falls back to DEFAULT_SETTINGS.llm_model when encryptedAgentSettings %s",
      (_, encryptedAgentSettings) => {
        expect(
          getModelFrom({
            settings: DEFAULT_SETTINGS,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            encryptedAgentSettings: encryptedAgentSettings as any,
          }),
        ).toBe(DEFAULT_SETTINGS.llm_model);
      },
    );
  });
});

describe("getDefaultConversationTitle", () => {
  it("formats the title using the first 5 characters of the conversation id", () => {
    expect(getDefaultConversationTitle("372eb-1234-5678-9abc")).toBe(
      "Conversation 372eb",
    );
  });
});

describe("toAppConversation", () => {
  const baseInfo: DirectConversationInfo = {
    id: "372eb-1234-5678-9abc",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };

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
      current_model_name: "Claude Sonnet 4.6",
      agent: {
        kind: "ACPAgent",
        acp_model: "claude-opus-4-7",
        llm: { model: "acp-managed" },
      },
    });
    expect(result.agent_kind).toBe("acp");
    expect(result.llm_model).toBe("Claude Sonnet 4.6");
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
  afterEach(() => {
    vi.unstubAllEnvs();
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_RUNTIME_SERVICES_INFO__;
  });

  it("returns undefined when VITE_RUNTIME_SERVICES_INFO is unset", () => {
    expect(buildRuntimeServicesSystemSuffix()).toBeUndefined();
  });

  it("returns undefined when the env var is malformed JSON", () => {
    vi.stubEnv("VITE_RUNTIME_SERVICES_INFO", "{not valid json");
    expect(buildRuntimeServicesSystemSuffix()).toBeUndefined();
  });

  it("returns undefined when the JSON has no services", () => {
    vi.stubEnv("VITE_RUNTIME_SERVICES_INFO", JSON.stringify({ mode: "x" }));
    expect(buildRuntimeServicesSystemSuffix()).toBeUndefined();
  });

  it("renders a <RUNTIME_SERVICES> block when an automation entry is present", () => {
    vi.stubEnv(
      "VITE_RUNTIME_SERVICES_INFO",
      JSON.stringify({
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
      }),
    );
    const suffix = buildRuntimeServicesSystemSuffix();
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
    vi.stubEnv(
      "VITE_RUNTIME_SERVICES_INFO",
      JSON.stringify({
        mode: "dev:safe",
        services: {
          agent_server: { url_from_agent: "http://localhost:18000" },
        },
      }),
    );
    const suffix = buildRuntimeServicesSystemSuffix();
    expect(suffix).toBeDefined();
    expect(suffix).toContain(
      "In particular, http://localhost:18000 inside your sandbox is the Agent Server",
    );
    expect(suffix).not.toContain(
      "In particular, http://localhost:8000 inside your sandbox",
    );
  });

  it("renders the frontend entry with the new key", () => {
    vi.stubEnv(
      "VITE_RUNTIME_SERVICES_INFO",
      JSON.stringify({
        mode: "dev:static",
        services: {
          agent_server: { url_from_agent: "http://localhost:18000" },
          frontend: {
            kind: "static",
            description: "Static-file server hosting the agent-canvas build.",
            url_from_agent: "http://localhost:3001",
          },
        },
      }),
    );
    const suffix = buildRuntimeServicesSystemSuffix();
    expect(suffix).toContain("* Frontend: http://localhost:3001");
    expect(suffix).toContain("Static-file server");
    // Should NOT mislabel a static-build frontend as "Vite frontend".
    expect(suffix).not.toContain("Vite frontend");
  });

  it("explicitly mentions when automation is absent", () => {
    vi.stubEnv(
      "VITE_RUNTIME_SERVICES_INFO",
      JSON.stringify({
        mode: "dev:safe",
        services: {
          agent_server: { url_from_agent: "http://localhost:18000" },
        },
      }),
    );
    const suffix = buildRuntimeServicesSystemSuffix();
    expect(suffix).toBeDefined();
    expect(suffix).toContain("Automation backend: not running");
  });

  it("falls back to window.__AGENT_CANVAS_RUNTIME_SERVICES_INFO__ when the env var is unset (static builds)", () => {
    // Static builds (Docker image / published binary) have no
    // VITE_RUNTIME_SERVICES_INFO baked in; scripts/static-server.mjs injects
    // the JSON onto window at serve time instead.
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_RUNTIME_SERVICES_INFO__ = JSON.stringify({
      mode: "docker",
      services: {
        agent_server: { url_from_agent: "http://127.0.0.1:18000" },
        automation: {
          url_from_agent: "http://127.0.0.1:8000",
          api_prefix: "/api/automation",
          auth_env_var: "OPENHANDS_AUTOMATION_API_KEY",
        },
      },
    });
    const suffix = buildRuntimeServicesSystemSuffix();
    expect(suffix).toBeDefined();
    expect(suffix).toContain("<RUNTIME_SERVICES>");
    expect(suffix).toContain("docker");
    expect(suffix).toContain("http://127.0.0.1:18000");
    expect(suffix).toContain("http://127.0.0.1:8000");
    expect(suffix).toContain(
      "X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY",
    );
  });

  it("prefers VITE_RUNTIME_SERVICES_INFO over the window fallback", () => {
    vi.stubEnv(
      "VITE_RUNTIME_SERVICES_INFO",
      JSON.stringify({
        mode: "dev:env",
        services: {
          agent_server: { url_from_agent: "http://localhost:18000" },
        },
      }),
    );
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_RUNTIME_SERVICES_INFO__ = JSON.stringify({
      mode: "docker:window",
      services: {
        agent_server: { url_from_agent: "http://127.0.0.1:99999" },
      },
    });
    const suffix = buildRuntimeServicesSystemSuffix();
    expect(suffix).toContain("dev:env");
    expect(suffix).not.toContain("docker:window");
    expect(suffix).not.toContain("99999");
  });
});

describe("agent_settings runtime services suffix", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("does not set system_message_suffix when no runtime info is provided", () => {
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
      query: "hello",
    }) as {
      agent_settings: { agent_context: Record<string, unknown> };
    };
    expect(payload.agent_settings.agent_context).toMatchObject({
      load_public_skills: false,
      load_user_skills: true,
      load_project_skills: true,
    });
    expect(Array.isArray(payload.agent_settings.agent_context.skills)).toBe(
      true,
    );
  });

  it("sets system_message_suffix when runtime info is provided", () => {
    vi.stubEnv(
      "VITE_RUNTIME_SERVICES_INFO",
      JSON.stringify({
        mode: "dev:automation",
        services: {
          agent_server: { url_from_agent: "http://localhost:18000" },
          automation: {
            url_from_agent: "http://localhost:18001",
          },
        },
      }),
    );
    const payload = buildStartConversationRequest({
      settings: DEFAULT_SETTINGS,
      query: "hello",
    }) as {
      agent_settings: { agent_context: Record<string, unknown> };
    };
    expect(payload.agent_settings.agent_context).toMatchObject({
      load_public_skills: false,
      load_user_skills: true,
    });
    expect(
      payload.agent_settings.agent_context.system_message_suffix as string,
    ).toContain("<RUNTIME_SERVICES>");
  });
});

describe("buildStartConversationRequest — ACP discriminator", () => {
  it("builds ACP agent settings when agent_kind is 'acp'", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_model: "claude-opus-4-5",
          // These fields are LLM-only and must NOT leak into ACP settings.
          // (mcp_config is handled separately — it IS forwarded for ACP; see
          // the dedicated tests below.)
          agent: "CodeActAgent",
          llm: { model: "gpt-4", api_key: "should-not-appear" },
          condenser: { enabled: true, max_size: 240 },
        },
      },
    }) as {
      agent?: unknown;
      agent_settings: Record<string, unknown> & {
        acp_command?: string[];
        acp_model?: string | null;
        agent_context?: unknown;
      };
      tags?: Record<string, string>;
    };

    expect(payload.agent).toBeUndefined();
    expect(payload.agent_settings.agent_kind).toBe("acp");
    expect(payload.agent_settings.acp_command).toEqual([
      "npx",
      "-y",
      "@agentclientprotocol/claude-agent-acp@0.44.0",
    ]);
    expect(payload.agent_settings.acp_model).toBe("claude-opus-4-5");
    // LLM-only fields must not leak into the ACP settings payload.
    expect(payload.agent_settings.llm).toBeUndefined();
    expect(payload.agent_settings.condenser).toBeUndefined();
    expect(payload.agent_settings.tools).toBeUndefined();
    const acpAgentContext = payload.agent_settings.agent_context as Record<
      string,
      unknown
    >;
    expect(acpAgentContext).toMatchObject({
      load_public_skills: false,
      load_user_skills: true,
      load_project_skills: true,
    });
    expect(Array.isArray(acpAgentContext.skills)).toBe(true);
    expect(payload.tags).toEqual({ [ACP_SERVER_TAG_KEY]: "claude-code" });
  });

  it("forwards mcp_config to the ACP subprocess when servers are configured", () => {
    // mcp_config is a shared field: the SDK's ACPAgent forwards these servers
    // to the ACP subprocess at session creation, so the start payload must
    // carry it (it is intentionally NOT one of the stripped ACP-only fields).
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
          mcp_config: {
            fetch: { command: "uvx", args: ["mcp-server-fetch"] },
          },
        },
      },
    }) as { agent_settings: { mcp_config?: unknown } };

    expect(payload.agent_settings.mcp_config).toEqual({
      fetch: { command: "uvx", args: ["mcp-server-fetch"] },
    });
  });

  it("omits mcp_config from the ACP payload when it carries no servers", () => {
    // An empty / serverless mcp_config must not be sent as ``mcp_config: {}``.
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
          mcp_config: {},
        },
      },
    }) as { agent_settings: { mcp_config?: unknown } };

    expect(payload.agent_settings.mcp_config).toBeUndefined();
  });

  it("does not include ACP-only fields in OpenHands agent settings", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...DEFAULT_SETTINGS.agent_settings,
          agent_kind: "openhands",
          llm: { model: "gpt-4" },
          acp_command: ["npx", "leftover"],
          acp_server: "claude-code",
        },
      },
    }) as {
      agent?: unknown;
      agent_settings: Record<string, unknown> & {
        llm: Record<string, unknown>;
      };
      tags?: Record<string, string>;
    };

    expect(payload.agent).toBeUndefined();
    expect(payload.agent_settings.agent_kind).toBe("openhands");
    expect(payload.agent_settings.acp_command).toBeUndefined();
    expect(payload.agent_settings.acp_server).toBeUndefined();
    expect(payload.agent_settings.llm.model).toBe("gpt-4");
    expect(payload.tags).toBeUndefined();
  });

  it("omits acp_model when the user clears it (null)", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "custom",
          acp_command: ["./bin/my-agent"],
          acp_model: null,
        },
      },
    }) as { agent_settings: Record<string, unknown> };

    expect(payload.agent_settings.agent_kind).toBe("acp");
    expect(payload.agent_settings.acp_model).toBeUndefined();
  });

  it("falls back to the preferred (Vertex-safe) default for a null Gemini acp_model", () => {
    // The start-request fallback is the third default-model surface (after
    // onboarding and Settings → Agent) — all three must substitute the same
    // Vertex-safe model, or a saved ``null`` would silently run gemini-cli's
    // 404-prone default on Vertex (software-agent-sdk#3532).
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "gemini-cli",
          acp_model: null,
        },
      },
    }) as { agent_settings: Record<string, unknown> };

    expect(payload.agent_settings.acp_model).toBe(ACP_VERTEX_SAFE_MODEL);
  });

  it("resolves an empty acp_command from the registry by acp_server", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_model: null,
        },
      },
    }) as {
      agent_settings: Record<string, unknown> & { acp_command?: unknown[] };
    };

    expect(payload.agent_settings.acp_command).toEqual([
      "npx",
      "-y",
      "@agentclientprotocol/claude-agent-acp@0.44.0",
    ]);
  });

  it("resolves an absent acp_command for built-in providers too", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "codex",
          acp_model: null,
        },
      },
    }) as {
      agent_settings: Record<string, unknown> & { acp_command?: unknown[] };
    };

    expect(payload.agent_settings.acp_command).toEqual([
      "npx",
      "-y",
      "@agentclientprotocol/codex-acp@1.1.2",
    ]);
  });

  it("leaves acp_command alone when acp_server is 'custom'", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "custom",
          acp_command: [],
          acp_model: null,
        },
      },
    }) as {
      agent_settings: Record<string, unknown> & { acp_command?: unknown[] };
    };

    expect(payload.agent_settings.acp_command).toEqual([]);
  });

  it("leaves acp_command alone for an unknown acp_server key", () => {
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "future-provider-not-yet-mirrored",
          acp_command: [],
          acp_model: null,
        },
      },
    }) as {
      agent_settings: Record<string, unknown> & { acp_command?: unknown[] };
    };

    expect(payload.agent_settings.acp_command).toEqual([]);
  });

  it("seeds the provider default when settings contains an empty acp_model", () => {
    // The form may carry an empty string after a user clears the model
    // input. Older behavior left ``acp_model`` absent and relied on the
    // agent-server's own default; the registry-default path
    // (resolveEffectiveAcpModel) is now authoritative on Canvas's side,
    // so an empty string resolves to the provider's ``default_model``
    // before the request leaves the client. Keeps the displayed Settings
    // → Agent default in sync with what the runtime actually starts.
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_command: [],
          acp_model: "",
        },
      },
    }) as {
      agent_settings: Record<string, unknown> & { acp_model?: unknown };
    };

    expect(payload.agent_settings.acp_model).toBe("opus[1m]");
  });

  it("omits acp_model for the custom preset when none is configured", () => {
    // The Custom preset has no registered ``default_model``, so an empty
    // ``acp_model`` falls through to ``undefined`` — the agent-server then
    // applies its own default. Distinct from the built-in providers
    // which substitute their registry default.
    const payload = buildStartConversationRequest({
      settings: {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          schema_version: 1,
          agent_kind: "acp",
          acp_server: "custom",
          acp_command: ["my-custom-acp"],
          acp_model: "",
        },
      },
    }) as {
      agent_settings: Record<string, unknown> & { acp_model?: unknown };
    };

    expect(payload.agent_settings.acp_model).toBeUndefined();
  });

  it("ACP → OpenHands → ACP round trip leaves no field leakage", () => {
    const baseAcpSettings = {
      ...DEFAULT_SETTINGS,
      agent_settings: {
        schema_version: 1,
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: [],
        // Legacy persisted value: provider creds no longer ride acp_env —
        // they flow through the Secrets panel (request.secrets). A stale
        // acp_env left on saved settings must be dropped, not forwarded.
        acp_env: { ANTHROPIC_API_KEY: "user-set-via-api" },
        acp_model: "claude-opus-4-5",
        agent: "CodeActAgent",
        llm: { model: "gpt-4o", api_key: "stale-from-prior-oh-run" },
        condenser: { enabled: true, max_size: 200 },
      },
    };

    const ohPayload = buildStartConversationRequest({
      settings: {
        ...baseAcpSettings,
        agent_settings: {
          ...baseAcpSettings.agent_settings,
          agent_kind: "openhands",
        },
      },
    }) as {
      agent_settings: Record<string, unknown> & {
        llm: Record<string, unknown>;
      };
    };

    expect(ohPayload.agent_settings.agent_kind).toBe("openhands");
    expect(ohPayload.agent_settings.acp_command).toBeUndefined();
    expect(ohPayload.agent_settings.acp_env).toBeUndefined();
    expect(ohPayload.agent_settings.acp_model).toBeUndefined();
    expect(ohPayload.agent_settings.acp_server).toBeUndefined();
    expect(ohPayload.agent_settings.llm.model).toBe("gpt-4o");

    const acpPayload = buildStartConversationRequest({
      settings: baseAcpSettings,
    }) as {
      agent_settings: Record<string, unknown> & {
        acp_command?: unknown;
        acp_env?: unknown;
        acp_model?: unknown;
        llm?: unknown;
        condenser?: unknown;
      };
    };

    expect(acpPayload.agent_settings.agent_kind).toBe("acp");
    expect(acpPayload.agent_settings.acp_command).toEqual([
      "npx",
      "-y",
      "@agentclientprotocol/claude-agent-acp@0.44.0",
    ]);
    expect(acpPayload.agent_settings.acp_model).toBe("claude-opus-4-5");
    // acp_env is no longer a forwarded ACP setting — a stale value on saved
    // settings is dropped rather than leaked into the conversation request.
    expect(acpPayload.agent_settings.acp_env).toBeUndefined();
    expect(acpPayload.agent_settings.llm).toBeUndefined();
    expect(acpPayload.agent_settings.condenser).toBeUndefined();
  });
});
