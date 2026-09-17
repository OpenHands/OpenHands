import {
  ConversationClient,
  FileClient,
  ProfilesClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { LOCAL_PLANNER_PARENT_TAG_KEY } from "#/utils/plan-file";

const {
  mockHttpGet,
  mockHttpPost,
  mockHttpDelete,
  mockConversationClient,
  mockFileClient,
  mockSettingsClient,
  mockSwitchProfile,
  mockSwitchLLM,
  mockGetSettings,
  mockGetSettingsForConversation,
  mockGetProfile,
  mockActivateProfile,
  mockListProfiles,
  mockGetTelemetryDistinctId,
  mockLoadHooks,
} = vi.hoisted(() => ({
  mockHttpGet: vi.fn(),
  mockHttpPost: vi.fn(),
  mockHttpDelete: vi.fn(),
  mockConversationClient: vi.fn(),
  mockFileClient: vi.fn(),
  mockSettingsClient: vi.fn(),
  mockSwitchProfile: vi.fn(),
  mockSwitchLLM: vi.fn(),
  mockGetSettings: vi.fn(),
  mockGetSettingsForConversation: vi.fn(),
  mockGetProfile: vi.fn(),
  mockActivateProfile: vi.fn(),
  mockListProfiles: vi.fn(),
  mockGetTelemetryDistinctId: vi.fn(),
  mockLoadHooks: vi.fn(),
}));

const originalFetch = global.fetch;
const fetchMock = vi.fn();

const localBackend: Backend = {
  id: "self-hosted",
  name: "Self-hosted",
  host: "http://localhost:54928",
  apiKey: "test-api-key",
  kind: "local",
};

vi.mock("@openhands/typescript-client/clients", async () => {
  const actual = await vi.importActual<
    typeof import("@openhands/typescript-client/clients")
  >("@openhands/typescript-client/clients");
  return {
    ...actual,
    ConversationClient: vi.fn(function ConversationClientMock() {
      return mockConversationClient();
    }),
    FileClient: vi.fn(function FileClientMock() {
      return mockFileClient();
    }),
    ProfilesClient: vi.fn(function ProfilesClientMock() {
      return {
        getProfile: mockGetProfile,
        activateProfile: mockActivateProfile,
        listProfiles: mockListProfiles,
      };
    }),
    SettingsClient: vi.fn(function SettingsClientMock() {
      return mockSettingsClient();
    }),
    VSCodeClient: vi.fn(function VSCodeClientMock() {
      return { getUrl: vi.fn() };
    }),
    HooksClient: vi.fn(function HooksClientMock() {
      return { loadHooks: mockLoadHooks };
    }),
  };
});

vi.mock("#/api/agent-server-config", () => ({
  DEFAULT_WORKING_DIR: "workspace/project",
  getAgentServerBaseUrl: vi.fn(() => "http://localhost:54928"),
  getAgentServerSessionApiKey: vi.fn(() => "test-api-key"),
  getAgentServerWorkingDir: vi.fn(() => "/workspace/project/agent-canvas"),
  getWorkspaceRootForBackend: vi.fn(() => "/workspace/project/agent-canvas"),
  buildConversationWorkingDirForBackend: vi.fn(
    (id: string) => `/state/workspaces/${id.replace(/-/g, "")}`,
  ),
  getAgentServerHeaders: vi.fn(() => ({ "X-Session-API-Key": "test-api-key" })),
  shouldLoadPublicSkills: vi.fn(() => true),
  syncBakedSessionApiKey: vi.fn(),
  getLockedCloudHost: vi.fn(() => null),
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    getSettings: mockGetSettings,
    getSettingsForConversation: mockGetSettingsForConversation,
  },
}));

vi.mock("#/services/telemetry", () => ({
  getTelemetryDistinctId: mockGetTelemetryDistinctId,
}));

describe("AgentServerConversationService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockHttpGet.mockReset();
    mockHttpPost.mockReset();
    mockHttpDelete.mockReset();
    mockGetProfile.mockReset();
    mockActivateProfile.mockReset();
    mockListProfiles.mockReset().mockResolvedValue({
      profiles: [],
      active_profile: null,
    });
    mockSwitchProfile.mockReset();
    mockSwitchLLM.mockReset();
    fetchMock.mockReset();
    global.fetch = originalFetch;
    vi.mocked(ConversationClient).mockClear();
    vi.mocked(FileClient).mockClear();
    vi.mocked(ProfilesClient).mockClear();
    vi.mocked(SettingsClient).mockClear();

    mockConversationClient.mockReturnValue({
      createConversation: async (payload: unknown) => {
        const response = await mockHttpPost("/api/conversations", payload);
        return response.data;
      },
      getConversations: async (conversationIds: string[]) => {
        const response = await mockHttpGet("/api/conversations", {
          params: { ids: conversationIds },
        });
        return response.data;
      },
      deleteConversation: async (conversationId: string) => {
        const response = await mockHttpDelete(
          `/api/conversations/${conversationId}`,
        );
        return response.data;
      },
      searchConversations: vi.fn(),
      getConversation: vi.fn(),
      sendEvent: vi.fn(),
      updateConversation: vi.fn(),
      switchProfile: mockSwitchProfile,
      switchLLM: mockSwitchLLM,
    });
    mockFileClient.mockReturnValue({
      downloadTextFile: async (path: string) => {
        const response = await mockHttpGet("/api/file/download", {
          params: { path },
          responseType: "arrayBuffer",
        });
        return new TextDecoder().decode(response.data);
      },
      downloadTrajectory: async (conversationId: string) => {
        const response = await mockHttpGet(
          `/api/file/download-trajectory/${conversationId}`,
          { responseType: "blob" },
        );
        return response.data;
      },
      // @spec WUP-001 — createConversation resolves relative working dirs
      // via FileClient.getHome before sending the conversation-start payload.
      getHome: async () => ({ home: "/Users/agent" }),
    });
    mockSettingsClient.mockReturnValue({
      listSecrets: vi.fn().mockResolvedValue({ secrets: [] }),
    });
  });

  describe("resilient conversation list parsing", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      setRegisteredBackends([localBackend]);
      setActiveSelection({ backendId: localBackend.id });
    });

    afterEach(() => {
      vi.restoreAllMocks();
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("drops a list item without an id and keeps its siblings", async () => {
      const searchConversations = vi.fn().mockResolvedValue({
        items: [
          { id: "good-before" },
          { title: "missing id" },
          { id: "good-after" },
        ],
        next_page_id: "next-page",
      });
      mockConversationClient.mockReturnValue({ searchConversations });
      const warn = vi
        .spyOn(console, "warn")
        .mockImplementation(() => undefined);

      const result =
        await AgentServerConversationService.searchConversations();

      expect(result.items.map(({ id }) => id)).toEqual([
        "good-before",
        "good-after",
      ]);
      expect(result.next_page_id).toBe("next-page");
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining("index 1 (id: <unknown>)"),
      );
      expect(warn.mock.calls[0]?.[0]).toContain(
        "Unable to load conversations because the selected agent server returned",
      );
    });

    it("drops a valid-id item when stats conversion fails", async () => {
      const searchConversations = vi.fn().mockResolvedValue({
        items: [
          { id: "good-before" },
          {
            id: "bad-stats",
            stats: { usage_to_metrics: { broken: null } },
          },
          { id: "good-after" },
        ],
      });
      mockConversationClient.mockReturnValue({ searchConversations });
      const warn = vi
        .spyOn(console, "warn")
        .mockImplementation(() => undefined);

      const result =
        await AgentServerConversationService.searchConversations();

      expect(result.items.map(({ id }) => id)).toEqual([
        "good-before",
        "good-after",
      ]);
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining("index 1 (id: bad-stats)"),
      );
      expect(warn.mock.calls[0]?.[0]).toContain("accumulated_cost");
    });

    it("returns an empty page when every list item is malformed", async () => {
      const searchConversations = vi.fn().mockResolvedValue({
        items: [null, { id: "   " }, []],
        next_page_id: null,
      });
      mockConversationClient.mockReturnValue({ searchConversations });

      await expect(
        AgentServerConversationService.searchConversations(),
      ).resolves.toEqual({ items: [], next_page_id: null });
    });

    it.each([
      ["primitive page", null],
      ["missing item list", { items: null, next_page_id: null }],
    ])(
      "still rejects a malformed search page shape: %s",
      async (_label, page) => {
        const searchConversations = vi.fn().mockResolvedValue(page);
        mockConversationClient.mockReturnValue({ searchConversations });

        await expect(
          AgentServerConversationService.searchConversations(),
        ).rejects.toThrow(
          "Unable to load conversations because the selected agent server returned",
        );
      },
    );

    it("filters planner-parent records without warning or counting a drop", async () => {
      const searchConversations = vi.fn().mockResolvedValue({
        items: [
          { id: "visible-before" },
          {
            id: "planner-helper",
            tags: { [LOCAL_PLANNER_PARENT_TAG_KEY]: "parent-conversation" },
          },
          { id: "visible-after" },
        ],
      });
      mockConversationClient.mockReturnValue({ searchConversations });
      const warn = vi
        .spyOn(console, "warn")
        .mockImplementation(() => undefined);

      const result =
        await AgentServerConversationService.searchConversations();

      expect(result.items.map(({ id }) => id)).toEqual([
        "visible-before",
        "visible-after",
      ]);
      expect(warn).not.toHaveBeenCalled();
    });

    it("keeps single-conversation and batch fetches strict", async () => {
      const corruptConversation = { title: "missing id" };
      const getConversation = vi.fn().mockResolvedValue(corruptConversation);
      const getConversations = vi
        .fn()
        .mockResolvedValue([corruptConversation]);
      mockConversationClient.mockReturnValue({
        getConversation,
        getConversations,
      });

      await expect(
        AgentServerConversationService.getRuntimeConversation(
          "corrupt-conversation",
          null,
        ),
      ).rejects.toThrow(
        "Unable to load conversations because the selected agent server returned",
      );
      await expect(
        AgentServerConversationService.batchGetAppConversations([
          "corrupt-conversation",
        ]),
      ).rejects.toThrow(
        "Unable to load conversations because the selected agent server returned",
      );
    });
  });

  describe("createConversation", () => {
    it("forwards the Canvas telemetry identity to the local agent server", async () => {
      mockGetTelemetryDistinctId.mockResolvedValue("ph-canvas-user");
      mockGetSettings.mockResolvedValue({
        agent_settings: { llm: { model: "gpt-4o" } },
        conversation_settings: {},
      });
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "gpt-4o" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockHttpPost.mockResolvedValue({
        data: {
          id: "conversation-1",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
        },
      });

      await AgentServerConversationService.createConversation();

      expect(mockHttpPost).toHaveBeenCalledWith(
        "/api/conversations",
        expect.objectContaining({ user_id: "ph-canvas-user" }),
      );
    });

    it("omits user_id when Canvas telemetry has no consented identity", async () => {
      mockGetTelemetryDistinctId.mockResolvedValue(null);
      mockGetSettings.mockResolvedValue({
        agent_settings: { llm: { model: "gpt-4o" } },
        conversation_settings: {},
      });
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "gpt-4o" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockHttpPost.mockResolvedValue({
        data: {
          id: "conversation-1",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
        },
      });

      await AgentServerConversationService.createConversation();

      const payload = mockHttpPost.mock.calls[0][1] as Record<string, unknown>;
      expect(payload).not.toHaveProperty("user_id");
    });

    it("passes the selected title profile to local conversation starts", async () => {
      mockGetSettings.mockResolvedValue({
        title_llm_profile: "Titles",
        agent_settings: { llm: { model: "gpt-4o" } },
        conversation_settings: {},
      });
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "gpt-4o" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockListProfiles.mockResolvedValue({
        profiles: [
          {
            name: "Titles",
            model: "anthropic/claude-haiku-3-5",
            base_url: null,
            api_key_set: true,
          },
        ],
        active_profile: null,
      });
      mockHttpPost.mockResolvedValue({
        data: {
          id: "ignored-server-id",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
        },
      });

      await AgentServerConversationService.createConversation();

      expect(mockHttpPost).toHaveBeenCalledWith(
        "/api/conversations",
        expect.objectContaining({ title_llm_profile: "Titles" }),
      );
    });

    // Regression for #16907 — the conversation's own `<workspace>/<hex>` dir
    // does not exist yet, so hooks looked up there are never found.
    it("looks project hooks up in the workspace root, not the conversation dir", async () => {
      mockGetSettings.mockResolvedValue({
        agent_settings: { llm: { model: "gpt-4o" } },
        conversation_settings: {},
      });
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "gpt-4o" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockHttpPost.mockResolvedValue({
        data: {
          id: "ignored-server-id",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
        },
      });

      await AgentServerConversationService.createConversation();

      const [payloadCall] = mockHttpPost.mock.calls;
      const payload = payloadCall[1] as {
        workspace: { working_dir: string };
      };
      expect(mockLoadHooks).toHaveBeenCalledWith({
        project_dir: "/workspace/project/agent-canvas",
      });
      expect(mockLoadHooks).not.toHaveBeenCalledWith({
        project_dir: payload.workspace.working_dir,
      });
    });

    // An explicit pick is the project, so hooks belong there.
    it("looks project hooks up in an explicitly picked workspace", async () => {
      mockGetSettings.mockResolvedValue({
        agent_settings: { llm: { model: "gpt-4o" } },
        conversation_settings: {},
      });
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "gpt-4o" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockHttpPost.mockResolvedValue({
        data: {
          id: "ignored-server-id",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
        },
      });

      await AgentServerConversationService.createConversation({
        workingDirOverride: "/Users/jane/projects/foo",
      });

      expect(mockLoadHooks).toHaveBeenCalledWith({
        project_dir: "/Users/jane/projects/foo",
      });
    });

    it("links a local conversation to its parent", async () => {
      mockGetSettings.mockResolvedValue({
        agent_settings: { llm: { model: "gpt-4o" } },
        conversation_settings: {},
      });
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "gpt-4o" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockHttpPost.mockResolvedValue({
        data: {
          id: "ignored-server-id",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
        },
      });

      await AgentServerConversationService.createConversation({
        workingDirOverride: "/Users/jane/projects/foo",
        workspaceMode: "new_worktree",
        parentConversationId: "parent-conversation-id",
      });

      const [payloadCall] = mockHttpPost.mock.calls;
      expect(payloadCall[1]).toMatchObject({
        parent_conversation_id: "parent-conversation-id",
      });
    });
  });

  describe("deleteConversation local branch", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("deletes the hidden planner helper reported by the server alongside its parent", async () => {
      mockHttpDelete.mockResolvedValue({ data: undefined });
      mockHttpGet.mockImplementation((_url: string, options: unknown) => {
        const ids = (options as { params: { ids: string[] } }).params.ids;
        if (ids.includes("conv-abc")) {
          return Promise.resolve({
            data: [
              {
                id: "conv-abc",
                created_at: "2024-01-01T00:00:00.000Z",
                updated_at: "2024-01-01T00:00:00.000Z",
                sub_conversation_ids: ["plan-abc"],
              },
            ],
          });
        }
        return Promise.resolve({
          data: [
            {
              id: "plan-abc",
              created_at: "2024-01-01T00:00:00.000Z",
              updated_at: "2024-01-01T00:00:00.000Z",
              tags: { plannerparent: "conv-abc" },
            },
          ],
        });
      });

      await AgentServerConversationService.deleteConversation("conv-abc");

      expect(mockHttpDelete).toHaveBeenCalledWith(
        "/api/conversations/plan-abc",
      );
      expect(mockHttpDelete).toHaveBeenCalledWith(
        "/api/conversations/conv-abc",
      );
    });

    it("still deletes the parent when the planner helper is already gone", async () => {
      mockHttpGet.mockImplementation((_url: string, options: unknown) => {
        const ids = (options as { params: { ids: string[] } }).params.ids;
        if (ids.includes("conv-abc")) {
          return Promise.resolve({
            data: [
              {
                id: "conv-abc",
                created_at: "2024-01-01T00:00:00.000Z",
                updated_at: "2024-01-01T00:00:00.000Z",
                sub_conversation_ids: ["plan-abc"],
              },
            ],
          });
        }
        return Promise.resolve({
          data: [
            {
              id: "plan-abc",
              created_at: "2024-01-01T00:00:00.000Z",
              updated_at: "2024-01-01T00:00:00.000Z",
              tags: { plannerparent: "conv-abc" },
            },
          ],
        });
      });
      mockHttpDelete.mockImplementation(async (path: string) => {
        if (path === "/api/conversations/plan-abc") {
          throw new Error("404 Not Found");
        }
        return { data: undefined };
      });

      await expect(
        AgentServerConversationService.deleteConversation("conv-abc"),
      ).resolves.toBeUndefined();

      expect(mockHttpDelete).toHaveBeenCalledWith(
        "/api/conversations/conv-abc",
      );
    });

    it("does not delete an unrelated non-planner child conversation", async () => {
      // Regression: sub_conversation_ids is the generic server-derived child
      // list, not a planner-only list — an untagged child (e.g. a delegated
      // sub-agent from another feature) must survive deleting the parent.
      mockHttpDelete.mockResolvedValue({ data: undefined });
      mockHttpGet.mockImplementation((_url: string, options: unknown) => {
        const ids = (options as { params: { ids: string[] } }).params.ids;
        if (ids.includes("conv-abc")) {
          return Promise.resolve({
            data: [
              {
                id: "conv-abc",
                created_at: "2024-01-01T00:00:00.000Z",
                updated_at: "2024-01-01T00:00:00.000Z",
                sub_conversation_ids: ["other-conv"],
              },
            ],
          });
        }
        return Promise.resolve({
          data: [
            {
              id: "other-conv",
              created_at: "2024-01-01T00:00:00.000Z",
              updated_at: "2024-01-01T00:00:00.000Z",
              tags: {},
            },
          ],
        });
      });

      await AgentServerConversationService.deleteConversation("conv-abc");

      expect(mockHttpDelete).not.toHaveBeenCalledWith(
        "/api/conversations/other-conv",
      );
      expect(mockHttpDelete).toHaveBeenCalledWith(
        "/api/conversations/conv-abc",
      );
    });
  });

  describe("createLocalPlanningConversation", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "openhands/global-model" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockHttpPost.mockResolvedValue({
        data: {
          id: "plan-abc",
          created_at: "2024-01-01T00:00:00.000Z",
          updated_at: "2024-01-01T00:00:00.000Z",
        },
      });
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("uses the configured working directory when the parent cannot be found", async () => {
      mockHttpGet.mockResolvedValue({ data: [] });
      await AgentServerConversationService.createLocalPlanningConversation(
        "missing-parent",
      );
      expect(mockHttpPost).toHaveBeenCalledWith(
        "/api/conversations",
        expect.objectContaining({
          workspace: expect.objectContaining({
            working_dir: "/workspace/project/agent-canvas",
          }),
        }),
      );
    });

    it("ignores active_profile for an ACP parent and falls back to global settings", async () => {
      // ACP parents get active_profile stamped with whatever LLM profile was
      // globally active at *their* creation time — not meaningfully tied to
      // the ACP agent — so it must not be treated as the planner's model.
      // agent_kind and active_profile are both derived client-side by
      // toAppConversation (from info.agent.kind and stored metadata
      // respectively), not read off the raw GET response.
      setStoredConversationMetadata("conv-abc", {
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        active_profile: "stale-acp-snapshot",
      });
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-abc",
            created_at: "2024-01-01T00:00:00.000Z",
            updated_at: "2024-01-01T00:00:00.000Z",
            agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
            sub_conversation_ids: [],
          },
        ],
      });

      await AgentServerConversationService.createLocalPlanningConversation(
        "conv-abc",
      );

      expect(mockGetProfile).not.toHaveBeenCalledWith("stale-acp-snapshot", {
        exposeSecrets: "encrypted",
      });
      const [, payload] = mockHttpPost.mock.calls[0] as [
        string,
        { agent: { llm: { model: string } } },
      ];
      expect(payload.agent.llm.model).toBe("openhands/global-model");
    });

    it("uses active_profile for an openhands-kind parent", async () => {
      // agent_kind and active_profile are both derived client-side by
      // toAppConversation (from info.agent.kind and stored metadata
      // respectively), not read off the raw GET response.
      setStoredConversationMetadata("conv-abc", {
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        active_profile: "switched-llm",
      });
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-abc",
            created_at: "2024-01-01T00:00:00.000Z",
            updated_at: "2024-01-01T00:00:00.000Z",
            sub_conversation_ids: [],
          },
        ],
      });
      mockGetProfile.mockResolvedValue({
        name: "switched-llm",
        api_key_set: true,
        config: { model: "openhands/switched-model" },
      });

      await AgentServerConversationService.createLocalPlanningConversation(
        "conv-abc",
      );

      expect(mockGetProfile).toHaveBeenCalledWith("switched-llm", {
        exposeSecrets: "encrypted",
      });
      const [, payload] = mockHttpPost.mock.calls[0] as [
        string,
        { agent: { llm: { model: string } } },
      ];
      expect(payload.agent.llm.model).toBe("openhands/switched-model");
      expect(getStoredConversationMetadata("conv-abc")).toMatchObject({
        active_profile: "switched-llm",
        local_planning_conversation_id: "plan-abc",
      });
    });

    it("inherits the launched agent profile model when no active LLM profile is stored", async () => {
      const { default: AgentProfilesService } =
        await import("#/api/agent-profiles-service/agent-profiles-service.api");
      const listProfiles = vi
        .spyOn(AgentProfilesService, "listProfiles")
        .mockResolvedValue({
          active_agent_profile_id: null,
          profiles: [
            {
              id: "launched-profile",
              name: "Parent agent",
              agent_kind: "openhands",
              revision: 1,
              llm_profile_ref: "parent-model",
              mcp_server_refs: [],
            },
          ],
        });
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-abc",
            created_at: "2024-01-01T00:00:00.000Z",
            updated_at: "2024-01-01T00:00:00.000Z",
            launched_agent_profile: {
              agent_profile_id: "launched-profile",
              revision: 1,
            },
            sub_conversation_ids: [],
          },
        ],
      });
      mockGetProfile.mockResolvedValue({
        name: "parent-model",
        api_key_set: true,
        config: { model: "openhands/parent-profile-model" },
      });
      try {
        await AgentServerConversationService.createLocalPlanningConversation(
          "conv-abc",
        );
        expect(mockGetProfile).toHaveBeenCalledWith("parent-model", {
          exposeSecrets: "encrypted",
        });
        expect(mockHttpPost).toHaveBeenCalledWith(
          "/api/conversations",
          expect.objectContaining({
            agent: expect.objectContaining({
              llm: expect.objectContaining({
                model: "openhands/parent-profile-model",
              }),
            }),
          }),
        );
      } finally {
        listProfiles.mockRestore();
      }
    });

    it("streams the planner's LLM tokens, matching the code agent", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-abc",
            created_at: "2024-01-01T00:00:00.000Z",
            updated_at: "2024-01-01T00:00:00.000Z",
            sub_conversation_ids: [],
          },
        ],
      });

      await AgentServerConversationService.createLocalPlanningConversation(
        "conv-abc",
      );

      const [, payload] = mockHttpPost.mock.calls[0] as [
        string,
        { agent: { llm: { stream?: boolean } } },
      ];
      expect(payload.agent.llm.stream).toBe(true);
    });
  });

  describe("conversation update fallbacks", () => {
    it("falls back to stats.usage_to_metrics when searchConversations omits metrics (#16480)", async () => {
      const searchSpy = vi.fn().mockResolvedValue({
        items: [
          {
            id: "conv-stats-only",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            stats: {
              usage_to_metrics: {
                default: {
                  model_name: "test-model",
                  accumulated_cost: 1.25,
                  max_budget_per_task: null,
                  accumulated_token_usage: {
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    context_window: 8000,
                    per_turn_token: 150,
                  },
                  costs: [],
                  response_latencies: [],
                  token_usages: [],
                },
              },
            },
          },
        ],
        next_page_id: null,
      });
      mockConversationClient.mockReturnValue({
        searchConversations: searchSpy,
      });

      const result =
        await AgentServerConversationService.searchConversations(10);

      expect(result.items[0]?.metrics?.accumulated_cost).toBe(1.25);
      expect(
        result.items[0]?.metrics?.accumulated_token_usage?.prompt_tokens,
      ).toBe(100);
    });

    it("preserves the launched Agent Profile through the wire normalizer", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-profile",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            launched_agent_profile: {
              agent_profile_id: "profile-1",
              revision: 3,
            },
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-profile",
        ]);

      expect(conversation?.launched_agent_profile).toEqual({
        agent_profile_id: "profile-1",
        revision: 3,
      });
    });

    it("carries well-formed wire tags through to AppConversation.tags", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-wire-tags",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
            tags: { acpserver: "claude-code", origin: "slack", owner: "alice" },
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-wire-tags",
        ]);

      expect(conversation?.tags).toEqual({
        acpserver: "claude-code",
        origin: "slack",
        owner: "alice",
      });
    });

    it("surfaces AppConversation.tags as null when the wire field is absent", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-no-tags",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-no-tags",
        ]);

      expect(conversation?.tags).toBeNull();
    });
  });
});
