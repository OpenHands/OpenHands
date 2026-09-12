import {
  ConversationClient,
  FileClient,
  ProfilesClient,
  SettingsClient,
  VSCodeClient,
} from "@openhands/typescript-client/clients";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { server } from "#/mocks/node";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import LLMSubscriptionService from "#/api/llm-subscription-service";
import {
  LLM_AUTH_TYPE_SUBSCRIPTION,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";

const {
  mockHttpGet,
  mockHttpPost,
  mockHttpDelete,
  mockConversationClient,
  mockFileClient,
  mockSettingsClient,
  mockSwitchProfile,
  mockSwitchLLM,
  mockSendEvent,
  mockGetConversation,
  mockSearchConversations,
  mockUpdateConversation,
  mockForkConversation,
  mockGetEvent,
  mockSwitchAcpModel,
  mockVSCodeGetUrl,
  mockGetSettings,
  mockGetSettingsForConversation,
  mockGetProfile,
  mockActivateProfile,
} = vi.hoisted(() => ({
  mockHttpGet: vi.fn(),
  mockHttpPost: vi.fn(),
  mockHttpDelete: vi.fn(),
  mockConversationClient: vi.fn(),
  mockFileClient: vi.fn(),
  mockSettingsClient: vi.fn(),
  mockSwitchProfile: vi.fn(),
  mockSwitchLLM: vi.fn(),
  mockSendEvent: vi.fn(),
  mockGetConversation: vi.fn(),
  mockSearchConversations: vi.fn(),
  mockUpdateConversation: vi.fn(),
  mockForkConversation: vi.fn(),
  mockGetEvent: vi.fn(),
  mockSwitchAcpModel: vi.fn(),
  mockVSCodeGetUrl: vi.fn(),
  mockGetSettings: vi.fn(),
  mockGetSettingsForConversation: vi.fn(),
  mockGetProfile: vi.fn(),
  mockActivateProfile: vi.fn(),
}));

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
      };
    }),
    SettingsClient: vi.fn(function SettingsClientMock() {
      return mockSettingsClient();
    }),
    VSCodeClient: vi.fn(function VSCodeClientMock() {
      return { getUrl: mockVSCodeGetUrl };
    }),
  };
});

vi.mock("#/api/agent-server-config", () => ({
  DEFAULT_WORKING_DIR: "workspace/project",
  getAgentServerBaseUrl: vi.fn(() => "http://localhost:54928"),
  getAgentServerSessionApiKey: vi.fn(() => "test-api-key"),
  getAgentServerWorkingDir: vi.fn(() => "/workspace/project/agent-canvas"),
  buildConversationWorkingDir: vi.fn(
    (id: string) => `/state/workspaces/${id.replace(/-/g, "")}`,
  ),
  buildConversationWorkingDirForBackend: vi.fn(
    (id: string) => `/state/workspaces/${id.replace(/-/g, "")}`,
  ),
  getWorkspaceRootForBackend: vi.fn(() => "/workspace/project/agent-canvas"),
  getAgentServerHeaders: vi.fn(() => ({ "X-Session-API-Key": "test-api-key" })),
  shouldLoadPublicSkills: vi.fn(() => true),
  syncBakedSessionApiKey: vi.fn(),
  getLockedCloudHost: vi.fn(() => null),
  getLockedCloudAuthMode: vi.fn(() => "api-key"),
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    getSettings: mockGetSettings,
    getSettingsForConversation: mockGetSettingsForConversation,
  },
}));

const localBackend: Backend = {
  id: "self-hosted",
  name: "Self-hosted",
  host: "http://localhost:54928",
  apiKey: "test-api-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "production",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const makeDirectConversation = (
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({
  id: "conv-1",
  title: "Conversation title",
  created_at: "2026-07-01T12:00:00.000Z",
  updated_at: "2026-07-01T12:01:00.000Z",
  execution_status: "idle",
  metrics: null,
  workspace: { working_dir: "/workspace/project/agent-canvas" },
  ...overrides,
});

const message = {
  role: "user" as const,
  content: [{ type: "text" as const, text: "Please inspect the project" }],
};

// The cloud path talks to the backend through the typescript-client's
// `CloudClient`, which uses `fetch` (direct calls to the cloud host, or a
// POST to the local `/api/cloud-proxy` for runtime-scoped calls). We capture
// those requests with MSW instead of asserting on a transport mock, so the
// tests exercise the real request-construction code in `CloudClient`.
interface RecordedRequest {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: unknown;
}

async function readJsonBody(request: Request): Promise<unknown> {
  const text = await request.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/**
 * Register MSW handlers for the given HTTP methods on any URL and record every
 * intercepted request. Each handler responds with `response` (JSON by default,
 * or the raw value when it is a `Blob`/`HttpResponse`). Returns the mutable
 * array of recorded requests so tests can assert on the real wire calls.
 */
function captureRequests(
  methods: Array<"get" | "post" | "patch" | "delete" | "put">,
  response: unknown = {},
): RecordedRequest[] {
  const recorded: RecordedRequest[] = [];
  const handlers = methods.map((method) =>
    http[method]("*", async ({ request }) => {
      recorded.push({
        method: request.method,
        url: request.url,
        headers: Object.fromEntries(request.headers.entries()),
        body: await readJsonBody(request),
      });
      if (response instanceof HttpResponse) return response;
      if (response instanceof Blob) return new HttpResponse(response);
      return HttpResponse.json(response as never);
    }),
  );
  server.use(...handlers);
  return recorded;
}

describe("AgentServerConversationService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockHttpGet.mockReset();
    mockHttpPost.mockReset();
    mockHttpDelete.mockReset();
    mockGetProfile.mockReset();
    mockActivateProfile.mockReset();
    mockSwitchProfile.mockReset();
    mockSwitchLLM.mockReset();
    mockSendEvent.mockReset();
    mockGetConversation.mockReset();
    mockSearchConversations.mockReset();
    mockUpdateConversation.mockReset();
    mockForkConversation.mockReset();
    mockGetEvent.mockReset();
    mockSwitchAcpModel.mockReset();
    mockVSCodeGetUrl.mockReset();
    vi.mocked(ConversationClient).mockClear();
    vi.mocked(FileClient).mockClear();
    vi.mocked(ProfilesClient).mockClear();
    vi.mocked(SettingsClient).mockClear();
    vi.mocked(VSCodeClient).mockClear();

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
      searchConversations: mockSearchConversations,
      getConversation: mockGetConversation,
      sendEvent: mockSendEvent,
      updateConversation: mockUpdateConversation,
      forkConversation: mockForkConversation,
      getEvent: mockGetEvent,
      switchAcpModel: mockSwitchAcpModel,
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

  describe("readConversationFile", () => {
    it("downloads the plan from the conversation's own working_dir when no filePath is provided", async () => {
      const encodedPlan = new TextEncoder().encode("# PLAN content").buffer;
      mockHttpGet.mockImplementation((url: string) => {
        if (url === "/api/conversations") {
          return Promise.resolve({
            data: [
              {
                id: "conv-123",
                created_at: "2024-01-01",
                updated_at: "2024-01-01",
                workspace: {
                  working_dir: "/workspace/project/agent-canvas/conv-123",
                },
              },
            ],
          });
        }
        return Promise.resolve({ data: encodedPlan });
      });

      const content =
        await AgentServerConversationService.readConversationFile("conv-123");

      expect(content).toBe("# PLAN content");
      expect(ConversationClient).toHaveBeenCalledWith({
        host: "http://localhost:54928",
        apiKey: "test-api-key",
        workingDir: "/workspace/project/agent-canvas",
      });
      expect(FileClient).toHaveBeenCalledWith({
        host: "http://localhost:54928",
        apiKey: "test-api-key",
        workingDir: "/workspace/project/agent-canvas",
      });
      expect(mockHttpGet).toHaveBeenCalledWith(
        "/api/file/download",
        expect.objectContaining({
          params: {
            path: "/workspace/project/agent-canvas/conv-123/.agents_tmp/PLAN.md",
          },
          responseType: "arrayBuffer",
        }),
      );
    });

    it("rejects explicit file paths outside the conversation workspace", async () => {
      mockHttpGet.mockImplementation((url: string) => {
        if (url === "/api/conversations") {
          return Promise.resolve({
            data: [
              {
                id: "conv-123",
                created_at: "2024-01-01",
                updated_at: "2024-01-01",
                workspace: {
                  working_dir: "/workspace/project/agent-canvas/conv-123",
                },
              },
            ],
          });
        }
        return Promise.resolve({ data: new ArrayBuffer(0) });
      });

      await expect(
        AgentServerConversationService.readConversationFile(
          "conv-123",
          "/workspace/project/agent-canvas/other/PLAN.md",
        ),
      ).rejects.toThrow(
        "Conversation file path must stay inside the workspace",
      );
      expect(mockHttpGet).not.toHaveBeenCalledWith(
        "/api/file/download",
        expect.anything(),
      );
    });
  });

  describe("createConversation", () => {
    it("generates a unique conversation_id and isolated working_dir per call", async () => {
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
      await AgentServerConversationService.createConversation();

      // The first conversation on a cold agent-server can exceed the client's
      // default timeout, so createConversation constructs its ConversationClient
      // with the extended CREATE_CONVERSATION_TIMEOUT_MS (5 minutes).
      expect(ConversationClient).toHaveBeenCalledWith({
        host: "http://localhost:54928",
        apiKey: "test-api-key",
        workingDir: "/workspace/project/agent-canvas",
        timeout: 300000,
      });
      expect(mockHttpPost).toHaveBeenCalledTimes(2);
      const [firstCall, secondCall] = mockHttpPost.mock.calls;
      const firstPayload = firstCall[1] as {
        conversation_id: string;
        workspace: { working_dir: string };
        worktree: boolean;
      };
      const secondPayload = secondCall[1] as {
        conversation_id: string;
        workspace: { working_dir: string };
        worktree: boolean;
      };

      expect(firstPayload.conversation_id).toBeTruthy();
      expect(secondPayload.conversation_id).toBeTruthy();
      expect(firstPayload.conversation_id).not.toBe(
        secondPayload.conversation_id,
      );
      const firstHex = firstPayload.conversation_id.replace(/-/g, "");
      const secondHex = secondPayload.conversation_id.replace(/-/g, "");
      expect(firstPayload.workspace.working_dir).toBe(
        `/state/workspaces/${firstHex}`,
      );
      expect(secondPayload.workspace.working_dir).toBe(
        `/state/workspaces/${secondHex}`,
      );
      expect(firstPayload.worktree).toBe(true);
      expect(secondPayload.worktree).toBe(true);
    });

    // @spec WUP-001 — When the default working_dir is relative, the
    // conversation-start payload must be anchored against the agent-server
    // home dir so the worktree and later file uploads agree on a writable
    // absolute path.
    it("resolves relative default working dirs against /api/file/home", async () => {
      const { buildConversationWorkingDirForBackend: mockedBuilder } =
        await import("#/api/agent-server-config");
      vi.mocked(mockedBuilder).mockImplementationOnce(
        (id: string) => `workspace/project/${id.replace(/-/g, "")}`,
      );
      const { clearAgentServerHomeDirCache } =
        await import("#/api/agent-server-home");
      clearAgentServerHomeDirCache();

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
        conversation_id: string;
        workspace: { working_dir: string };
      };
      const hex = payload.conversation_id.replace(/-/g, "");
      expect(payload.workspace.working_dir).toBe(
        `/Users/agent/workspace/project/${hex}`,
      );
    });

    // @spec WUP-001 — User-supplied workspace overrides are already absolute
    // (they come from `search_subdirs`), so they must pass through verbatim.
    it("leaves an absolute workingDirOverride untouched", async () => {
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

      const [payloadCall] = mockHttpPost.mock.calls;
      const payload = payloadCall[1] as {
        workspace: { working_dir: string };
        worktree: boolean;
      };
      expect(payload.workspace.working_dir).toBe("/Users/jane/projects/foo");
      expect(payload.worktree).toBe(false);
    });

    it("honors an explicit new-worktree mode for a selected workspace", async () => {
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
      });

      const [payloadCall] = mockHttpPost.mock.calls;
      const payload = payloadCall[1] as {
        workspace: { working_dir: string };
        worktree: boolean;
      };
      expect(payload.workspace.working_dir).toBe("/Users/jane/projects/foo");
      expect(payload.worktree).toBe(true);
    });
  });

  describe("downloadConversation local branch", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("hits the local /api/file/download-trajectory endpoint with responseType blob when active backend is local", async () => {
      const zipBlob = new Blob(["zip-bytes"], { type: "application/zip" });
      mockHttpGet.mockResolvedValue({ data: zipBlob });

      const result =
        await AgentServerConversationService.downloadConversation("conv-abc");

      expect(mockHttpGet).toHaveBeenCalledWith(
        "/api/file/download-trajectory/conv-abc",
        expect.objectContaining({ responseType: "blob" }),
      );
      expect(result).toBe(zipBlob);
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

    it("hits the local /api/conversations/{id} endpoint when active backend is local", async () => {
      mockHttpDelete.mockResolvedValue({ data: undefined });

      await AgentServerConversationService.deleteConversation("conv-abc");

      expect(mockHttpDelete).toHaveBeenCalledWith(
        "/api/conversations/conv-abc",
      );
    });
  });

  describe("conversation update fallbacks", () => {
    it("throws a useful error when repository update cannot reload the conversation", async () => {
      mockHttpGet.mockResolvedValue({ data: [] });

      await expect(
        AgentServerConversationService.updateConversationRepository(
          "missing-conv",
          "OpenHands/agent-canvas",
        ),
      ).rejects.toThrow("Conversation missing-conv was not found");
    });

    it("throws a useful error when title update cannot reload the conversation", async () => {
      mockHttpGet.mockResolvedValue({ data: [] });

      await expect(
        AgentServerConversationService.updateConversationTitle(
          "missing-conv",
          "New title",
        ),
      ).rejects.toThrow("Conversation missing-conv was not found");
    });

    it("normalizes conversation list items with missing timestamps", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-no-timestamps",
            title: "Conversation without timestamps",
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-no-timestamps",
        ]);

      expect(conversation).toMatchObject({
        id: "conv-no-timestamps",
        created_at: "1970-01-01T00:00:00.000Z",
        updated_at: "1970-01-01T00:00:00.000Z",
      });
    });

    it("throws a user-friendly error for unusable conversation list responses", async () => {
      mockHttpGet.mockResolvedValue({ data: [{ title: "missing id" }] });

      await expect(
        AgentServerConversationService.batchGetAppConversations(["missing-id"]),
      ).rejects.toThrow(
        "Unable to load conversations because the selected agent server returned",
      );
    });

    it("preserves sandbox_status from batchGetAppConversations response", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-paused",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            sandbox_status: "PAUSED",
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-paused",
        ]);

      expect(conversation?.sandbox_status).toBe("PAUSED");
    });

    it("preserves sandbox_status from searchConversations response", async () => {
      const searchSpy = vi.fn().mockResolvedValue({
        items: [
          {
            id: "conv-paused-search",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            sandbox_status: "PAUSED",
          },
        ],
        next_page_id: null,
      });
      // Only searchConversations is called by the service method under test,
      // so we don't need to reproduce the full client mock object.
      mockConversationClient.mockReturnValue({
        searchConversations: searchSpy,
      });

      const result =
        await AgentServerConversationService.searchConversations(10);

      expect(result.items[0]?.sandbox_status).toBe("PAUSED");
    });

    it("passes sandbox_status null through when field is absent", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-no-status",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-no-status",
        ]);

      expect(conversation?.sandbox_status).toBeNull();
    });

    it("sanitizes malformed optional conversation fields", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-malformed-fields",
            title: "Conversation with malformed fields",
            metrics: {
              accumulated_cost: "1.23",
              max_budget_per_task: 10,
              accumulated_token_usage: {
                prompt_tokens: "123",
                completion_tokens: 4,
              },
            },
            agent: "not an agent object",
            workspace: "not a workspace object",
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-malformed-fields",
        ]);

      expect(conversation?.metrics).toEqual({
        accumulated_cost: null,
        max_budget_per_task: 10,
        accumulated_token_usage: {
          prompt_tokens: 0,
          completion_tokens: 4,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          context_window: 0,
          per_turn_token: 0,
        },
      });
      expect(conversation?.llm_model).toBeTruthy();
      expect(conversation?.workspace?.working_dir).toBe(
        "/workspace/project/agent-canvas",
      );
    });

    it("preserves the new ACP model fields through the wire normalizer", async () => {
      // Direct adapter tests pass DirectConversationInfo objects in-process
      // and so can't catch the case where the wire-format normalizer
      // (``normalizeAgent`` + ``requireDirectConversationInfo``) drops the
      // newly-added ACP fields. Exercises the full HTTP -> AppConversation
      // path so the chip's model resolution actually has the inputs it
      // needs on a real local-backend fetch.
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-acp-model-wire",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            agent: {
              kind: "ACPAgent",
              acp_model: "claude-opus-4-7",
              llm: { model: "acp-managed" },
            },
            current_model_id: "claude-opus-4-7",
            current_model_name: "Claude Opus 4.7",
            tags: { acpserver: "claude-code" },
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-acp-model-wire",
        ]);

      // ``current_model_name`` wins the precedence chain in the adapter.
      expect(conversation?.agent_kind).toBe("acp");
      expect(conversation?.llm_model).toBe("Claude Opus 4.7");
    });

    it("sources acp_server from the agent when the acpserver tag is absent", async () => {
      // Profile launches don't stamp the ``acpserver`` tag client-side, so the
      // provider identity must survive from ``agent.acp_server`` (SDK #3692)
      // through ``normalizeAgent``. Without it the chip degrades to a generic
      // "ACP" and the in-conversation model picker shows no options (#1571).
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-acp-server-from-agent",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            agent: {
              kind: "ACPAgent",
              acp_server: "claude-code",
              acp_model: "claude-sonnet-4-5",
              llm: { model: "acp-managed" },
            },
            // No ``acpserver`` tag — mirrors an agent_profile_id launch.
            tags: {},
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-acp-server-from-agent",
        ]);

      expect(conversation?.agent_kind).toBe("acp");
      expect(conversation?.acp_server).toBe("claude-code");
    });

    it("falls back to acp_model when SDK runtime fields are absent on the wire", async () => {
      // Older agent-servers don't populate ``current_model_*``. The
      // adapter must still surface a model on the chip — falling through
      // to ``agent.acp_model`` (the Canvas-configured value).
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-acp-fallback",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            agent: {
              kind: "ACPAgent",
              acp_model: "claude-sonnet-4-6",
              llm: { model: "acp-managed" },
            },
            tags: { acpserver: "claude-code" },
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-acp-fallback",
        ]);

      expect(conversation?.llm_model).toBe("claude-sonnet-4-6");
    });

    it("extracts the acpserver tag from the wire payload for the sidebar chip", async () => {
      // The agent-server stamps ``tags.acpserver`` at conversation create
      // time (see ``buildStartConversationRequest``); the read path
      // must surface it so the conversation card can render the human
      // ACP-agent badge ("Claude Code" / "Codex" / "Gemini CLI").
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-acp",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
            tags: { acpserver: "claude-code" },
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-acp",
        ]);

      expect(conversation?.agent_kind).toBe("acp");
      expect(conversation?.acp_server).toBe("claude-code");
    });

    it("drops non-string tag values while preserving the well-typed ones", async () => {
      // The wire field is server-validated to ``Record[str, str]`` but a
      // misbehaving server (or a future schema drift) shouldn't crash the
      // parser — we drop non-string values and keep the rest so the
      // sidebar still gets whatever good keys made it through.
      mockHttpGet.mockResolvedValue({
        data: [
          {
            id: "conv-malformed-tags",
            created_at: "2024-01-01",
            updated_at: "2024-01-01",
            agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
            tags: {
              acpserver: "codex",
              numeric: 42,
              nested: { inner: "x" },
              listy: ["a", "b"],
              nully: null,
            },
          },
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-malformed-tags",
        ]);

      // ``acp_server`` is the surfaced field on AppConversation; tags is
      // only on DirectConversationInfo. Asserting both via this read
      // path keeps the test honest end-to-end.
      expect(conversation?.acp_server).toBe("codex");
    });
  });

  describe("switchProfile", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("switches an active conversation with the full encrypted profile config", async () => {
      mockGetProfile.mockResolvedValue({
        name: "haiku",
        config: {
          model: "openhands/claude-haiku-4-5",
          api_key: "encrypted-key",
        },
        api_key_set: true,
      });
      mockSwitchLLM.mockResolvedValue(undefined);

      await AgentServerConversationService.switchProfile("conv-1", "haiku");

      expect(mockGetProfile).toHaveBeenCalledWith("haiku", {
        exposeSecrets: "encrypted",
      });
      expect(mockSwitchLLM).toHaveBeenCalledWith(
        "conv-1",
        expect.objectContaining({
          model: "openhands/claude-haiku-4-5",
          api_key: "encrypted-key",
          // Streaming must stay enabled after a mid-conversation switch.
          stream: true,
          usage_id: expect.stringMatching(/^profile:haiku:/),
        }),
      );
      // Per-convo path: global default is left untouched and profile secrets are
      // only fetched as encrypted values for direct round-trip to switch_llm.
      expect(mockActivateProfile).not.toHaveBeenCalled();
      expect(mockSwitchProfile).not.toHaveBeenCalled();
    });

    it("surfaces encrypted profile export failures instead of using the stale profile switch path", async () => {
      const error = new Error("No cipher");
      mockGetProfile.mockRejectedValueOnce(error);

      await expect(
        AgentServerConversationService.switchProfile("conv-1", "haiku"),
      ).rejects.toThrow(error);

      expect(mockGetProfile).toHaveBeenCalledWith("haiku", {
        exposeSecrets: "encrypted",
      });
      expect(mockSwitchProfile).not.toHaveBeenCalled();
      expect(mockSwitchLLM).not.toHaveBeenCalled();
      expect(mockActivateProfile).not.toHaveBeenCalled();
    });

    it("activates the profile globally when called without a conversationId", async () => {
      mockActivateProfile.mockResolvedValue({
        name: "haiku",
        message: "ok",
        llm_applied: true,
      });

      await AgentServerConversationService.switchProfile(null, "haiku");

      expect(mockActivateProfile).toHaveBeenCalledWith("haiku");
      // Home-page path: don't touch any conversation's LLM.
      expect(mockGetProfile).not.toHaveBeenCalled();
      expect(mockSwitchProfile).not.toHaveBeenCalled();
      expect(mockSwitchLLM).not.toHaveBeenCalled();
    });

    it("routes a cloud conversation switch through the app-server switch_profile endpoint", async () => {
      const cloudBackend: Backend = {
        id: "prod",
        name: "Production",
        host: "https://app.all-hands.dev",
        apiKey: "bearer-token",
        kind: "cloud",
      };
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests = captureRequests(["post"], { success: true });

      await AgentServerConversationService.switchProfile("conv-1", "haiku");

      expect(requests).toHaveLength(1);
      const [request] = requests;
      expect(request.method).toBe("POST");
      expect(request.url).toBe(
        "https://app.all-hands.dev/api/v1/app-conversations/conv-1/switch_profile",
      );
      expect(request.body).toEqual({ profile_name: "haiku" });
      // Cloud resolves the swap server-side: no client-side encrypted profile
      // fetch and no direct switch_llm call.
      expect(mockGetProfile).not.toHaveBeenCalled();
      expect(mockSwitchLLM).not.toHaveBeenCalled();
    });
  });

  describe("cloud branches", () => {
    const cloudBackend: Backend = {
      id: "prod",
      name: "Production",
      host: "https://app.all-hands.dev",
      apiKey: "bearer-token",
      kind: "cloud",
    };

    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("forwards parent_conversation_id, agent_type, and sandbox_id to the cloud createConversation payload", async () => {
      // Arrange
      const requests = captureRequests(["post"], {
        id: "task-1",
        status: "WORKING",
        app_conversation_id: null,
        agent_server_url: null,
        request: {},
        created_at: "2024-01-01",
        updated_at: "2024-01-01",
      });

      // Act
      await AgentServerConversationService.createConversation({
        metadata: null,
        parentConversationId: "parent-conv-1",
        agentType: "plan",
        sandboxId: "sandbox-9",
      });

      // Assert
      expect(requests).toHaveLength(1);
      const [request] = requests;
      expect(request.method).toBe("POST");
      expect(request.url).toBe(
        `${cloudBackend.host}/api/v1/app-conversations`,
      );
      expect(request.headers.authorization).toBe("Bearer bearer-token");
      expect(request.body).toMatchObject({
        parent_conversation_id: "parent-conv-1",
        agent_type: "plan",
        sandbox_id: "sandbox-9",
      });
    });

    it("routes readConversationFile to the cloud file endpoint with the file_path query param", async () => {
      // Arrange
      const requests = captureRequests(["get"], "# PLAN content");

      // Act
      const content =
        await AgentServerConversationService.readConversationFile(
          "conv-cloud-1",
        );

      // Assert
      expect(content).toBe("# PLAN content");
      expect(requests).toHaveLength(1);
      const [request] = requests;
      expect(request.method).toBe("GET");
      expect(request.headers.authorization).toBe("Bearer bearer-token");
      expect(request.url).toBe(
        `${cloudBackend.host}/api/v1/app-conversations/conv-cloud-1/file?file_path=%2Fworkspace%2Fproject%2F.agents_tmp%2FPLAN.md`,
      );
    });
  });

  describe("conversation transport and backend routing", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      setRegisteredBackends([localBackend]);
      setActiveSelection({ backendId: localBackend.id });
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("sends a local message to the selected runtime and returns the accepted message", async () => {
      mockSendEvent.mockResolvedValue(undefined);

      const result = await AgentServerConversationService.sendMessage(
        "conv-1",
        message,
        {
          conversationUrl: "http://runtime.internal:9000",
          sessionApiKey: "runtime-key",
        },
      );

      expect(mockSendEvent).toHaveBeenCalledWith("conv-1", message, {
        run: true,
      });
      expect(ConversationClient).toHaveBeenCalledWith(
        expect.objectContaining({
          host: "http://runtime.internal:9000",
          apiKey: "runtime-key",
        }),
      );
      expect(result).toEqual(message);
    });

    it("sends a cloud message with runtime credentials supplied by the caller", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const proxyRequests = captureRequests(["post"], {});

      const result = await AgentServerConversationService.sendMessage(
        "conv-cloud",
        message,
        {
          conversationUrl:
            "http://runtime.example/api/conversations/conv-cloud",
          sessionApiKey: "session-key",
        },
      );

      // Runtime credentials supplied by the caller: no batch-get is issued,
      // only the single cloud-proxy POST to the runtime.
      expect(proxyRequests).toHaveLength(1);
      const [request] = proxyRequests;
      expect(request.url).toBe("http://localhost:54928/api/cloud-proxy");
      expect(request.body).toMatchObject({
        host: "http://runtime.example",
        method: "POST",
        path: "/api/conversations/conv-cloud/events",
        headers: { "X-Session-API-Key": "session-key" },
        body: { ...message, run: true },
      });
      expect(result).toEqual(message);
    });

    it("loads missing cloud runtime credentials before sending a message", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests: RecordedRequest[] = [];
      server.use(
        http.get(
          "https://app.all-hands.dev/api/v1/app-conversations",
          async ({ request }) => {
            requests.push({
              method: request.method,
              url: request.url,
              headers: Object.fromEntries(request.headers.entries()),
              body: await readJsonBody(request),
            });
            return HttpResponse.json([
              {
                id: "conv-cloud",
                conversation_url:
                  "  http://runtime.example/api/conversations/conv-cloud  ",
                session_api_key: "  fetched-key  ",
              },
            ]);
          },
        ),
        http.post("*/api/cloud-proxy", async ({ request }) => {
          requests.push({
            method: request.method,
            url: request.url,
            headers: Object.fromEntries(request.headers.entries()),
            body: await readJsonBody(request),
          });
          return HttpResponse.json({});
        }),
      );

      await AgentServerConversationService.sendMessage("conv-cloud", message);

      const batchGet = requests.find((req) => req.method === "GET");
      const proxyPost = requests.find(
        (req) => req.url === "http://localhost:54928/api/cloud-proxy",
      );
      expect(batchGet?.url).toContain("ids=conv-cloud");
      expect(proxyPost?.body).toMatchObject({
        host: "http://runtime.example",
        headers: { "X-Session-API-Key": "fetched-key" },
      });
    });

    it("explains when a cloud sandbox has not published runtime credentials", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const proxyRequests = captureRequests(["post"], {});
      server.use(
        http.get(
          "https://app.all-hands.dev/api/v1/app-conversations",
          () => HttpResponse.json([]),
        ),
      );

      await expect(
        AgentServerConversationService.sendMessage("conv-cloud", message),
      ).rejects.toThrow("Conversation sandbox is still starting");
      expect(proxyRequests).toHaveLength(0);
    });

    it("returns no start task for a synchronously-created local conversation", async () => {
      await expect(
        AgentServerConversationService.getStartTask("local-task"),
      ).resolves.toBeNull();
    });

    it("loads a cloud conversation start task", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const task = {
        id: "task-1",
        created_by_user_id: null,
        status: "READY" as const,
        detail: null,
        app_conversation_id: "conv-cloud",
        agent_server_url: "http://runtime.example",
        request: {},
        created_at: "2026-07-01T12:00:00.000Z",
        updated_at: "2026-07-01T12:01:00.000Z",
      };
      const requests = captureRequests(["get"], [task]);

      await expect(
        AgentServerConversationService.getStartTask("task-1"),
      ).resolves.toEqual(task);
      expect(requests).toHaveLength(1);
      expect(requests[0].url).toContain(
        "/api/v1/app-conversations/start-tasks?ids=task-1",
      );
    });

    it("requests a VS Code URL for the conversation workspace", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          makeDirectConversation({
            workspace: { working_dir: "/workspace/repos/canvas" },
          }),
        ],
      });
      mockVSCodeGetUrl.mockResolvedValue("http://localhost:3000/vscode");

      const result = await AgentServerConversationService.getVSCodeUrl(
        "conv-1",
        "http://runtime.internal:9000",
        "runtime-key",
      );

      expect(mockVSCodeGetUrl).toHaveBeenCalledWith({
        baseUrl: window.location.origin,
        workspaceDir: "/workspace/repos/canvas",
      });
      expect(result).toEqual({
        vscode_url: "http://localhost:3000/vscode",
      });
    });

    it("uses the configured working directory when a conversation has no workspace", async () => {
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ workspace: null })],
      });

      await expect(
        AgentServerConversationService.resolveConversationWorkingDir("conv-1"),
      ).resolves.toBe("/workspace/project/agent-canvas");
    });

    it("uses the configured working directory when the conversation is absent", async () => {
      mockHttpGet.mockResolvedValue({ data: [] });

      await expect(
        AgentServerConversationService.resolveConversationWorkingDir(
          "missing-conv",
        ),
      ).resolves.toBe("/workspace/project/agent-canvas");
    });

    it("omits the browser origin when requesting a VS Code URL during SSR", async () => {
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ workspace: null })],
      });
      mockVSCodeGetUrl.mockResolvedValue("http://localhost:3000/vscode");
      const browserWindow = window;
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: undefined,
      });

      try {
        await AgentServerConversationService.getVSCodeUrl("conv-1", undefined);
      } finally {
        Object.defineProperty(globalThis, "window", {
          configurable: true,
          value: browserWindow,
        });
      }

      expect(mockVSCodeGetUrl).toHaveBeenCalledWith({
        baseUrl: undefined,
        workspaceDir: "/workspace/project/agent-canvas",
      });
    });

    it("does not contact a backend for an empty conversation batch", async () => {
      await expect(
        AgentServerConversationService.batchGetAppConversations([]),
      ).resolves.toEqual([]);
      expect(ConversationClient).not.toHaveBeenCalled();
    });

    it("routes conversation batches to the cloud app endpoint", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const conversation = {
        id: "conv-cloud",
        title: "Cloud conversation",
      };
      const requests = captureRequests(["get"], [conversation]);

      await expect(
        AgentServerConversationService.batchGetAppConversations(["conv-cloud"]),
      ).resolves.toEqual([conversation]);
      expect(requests).toHaveLength(1);
      expect(requests[0].url).toContain("ids=conv-cloud");
    });

    it("rejects public sharing on a local backend", async () => {
      await expect(
        AgentServerConversationService.updateConversationPublicFlag(
          "conv-1",
          true,
        ),
      ).rejects.toThrow("Public sharing requires a cloud backend");
    });
  });

  describe("conversation response validation and runtime state", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      setRegisteredBackends([localBackend]);
      setActiveSelection({ backendId: localBackend.id });
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("rejects a conversation batch that is not a list", async () => {
      mockHttpGet.mockResolvedValue({ data: { id: "conv-1" } });

      await expect(
        AgentServerConversationService.batchGetAppConversations(["conv-1"]),
      ).rejects.toThrow(
        "Unable to load conversations because the selected agent server returned",
      );
    });

    it.each([
      ["null item", null],
      ["array item", []],
      ["numeric id", { id: 7 }],
      ["blank id", { id: "   " }],
    ])(
      "rejects a malformed conversation list item: %s",
      async (_label, item) => {
        mockHttpGet.mockResolvedValue({ data: [item] });

        await expect(
          AgentServerConversationService.batchGetAppConversations(["conv-1"]),
        ).rejects.toThrow(
          "Unable to load conversations because the selected agent server returned",
        );
      },
    );

    it("normalizes camel-case timestamps and malformed nested optional fields", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          makeDirectConversation({
            id: "  conv-camel  ",
            created_at: undefined,
            updated_at: "   ",
            createdAt: "2026-06-30T10:00:00.000Z",
            updatedAt: "2026-06-30T11:00:00.000Z",
            execution_status: 42,
            metrics: {
              accumulated_cost: 1.5,
              max_budget_per_task: "unlimited",
              accumulated_token_usage: null,
            },
            agent: {
              kind: 7,
              acp_server: 7,
              acp_model: 7,
              llm: "not-an-object",
            },
            workspace: { working_dir: 7 },
            tags: "not-an-object",
            current_model_id: 7,
            current_model_name: 7,
          }),
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-camel",
        ]);

      expect(conversation).toMatchObject({
        id: "conv-camel",
        created_at: "2026-06-30T10:00:00.000Z",
        updated_at: "1970-01-01T00:00:00.000Z",
        execution_status: "idle",
        metrics: {
          accumulated_cost: 1.5,
          max_budget_per_task: null,
          accumulated_token_usage: null,
        },
      });
    });

    it("accepts a legacy list response from conversation search", async () => {
      mockSearchConversations.mockResolvedValue([
        makeDirectConversation({ id: "legacy-conv" }),
      ]);

      const result = await AgentServerConversationService.searchConversations();

      expect(mockSearchConversations).toHaveBeenCalledWith({
        limit: 20,
        page_id: undefined,
        sort_order: "UPDATED_AT_DESC",
      });
      expect(result.items[0]?.id).toBe("legacy-conv");
      expect(result.next_page_id).toBeNull();
    });

    it("preserves a string cursor from a paginated conversation search", async () => {
      mockSearchConversations.mockResolvedValue({
        items: [makeDirectConversation({ id: "page-conv" })],
        next_page_id: "next-page",
      });

      const result = await AgentServerConversationService.searchConversations(
        5,
        "current-page",
      );

      expect(result.next_page_id).toBe("next-page");
      expect(mockSearchConversations).toHaveBeenCalledWith({
        limit: 5,
        page_id: "current-page",
        sort_order: "UPDATED_AT_DESC",
      });
    });

    it.each([
      ["primitive page", "invalid-page"],
      ["missing item list", { items: null, next_page_id: null }],
    ])(
      "rejects a malformed conversation search response: %s",
      async (_label, page) => {
        mockSearchConversations.mockResolvedValue(page);

        await expect(
          AgentServerConversationService.searchConversations(),
        ).rejects.toThrow(
          "Unable to load conversations because the selected agent server returned",
        );
      },
    );

    it("routes conversation search through the cloud backend", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests = captureRequests(["get"], {
        items: [],
        next_page_id: "cloud-next",
      });

      const result = await AgentServerConversationService.searchConversations(
        7,
        "cloud-page",
      );

      expect(result).toEqual({ items: [], next_page_id: "cloud-next" });
      expect(requests).toHaveLength(1);
      expect(requests[0].url).toContain(
        "limit=7&page_id=cloud-page&sort_order=UPDATED_AT_DESC",
      );
    });

    it("uses safe defaults for a sparse runtime response", async () => {
      mockGetConversation.mockResolvedValue(
        makeDirectConversation({
          title: "   ",
          execution_status: null,
          metrics: {
            accumulated_cost: null,
            max_budget_per_task: null,
            accumulated_token_usage: null,
          },
          stats: null,
        }),
      );

      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-1",
          undefined,
        );

      expect(result).toMatchObject({
        id: "conv-1",
        title: "Conversation conv-",
        status: "idle",
        stats: { usage_to_metrics: {} },
      });
    });

    it("preserves valid runtime status, title, metrics, and stats", async () => {
      const stats = { usage_to_metrics: { agent: { model_name: "gpt-5" } } };
      mockGetConversation.mockResolvedValue(
        makeDirectConversation({
          title: "Running conversation",
          execution_status: "running",
          metrics: {
            accumulated_cost: 2,
            max_budget_per_task: 10,
            accumulated_token_usage: {
              prompt_tokens: 5,
              completion_tokens: 4,
              cache_read_tokens: 3,
              cache_write_tokens: 2,
              context_window: 100,
              per_turn_token: 9,
            },
          },
          stats,
        }),
      );

      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-1",
          "http://runtime.internal:9000",
          "runtime-key",
        );

      expect(result).toMatchObject({
        title: "Running conversation",
        status: "running",
        metrics: {
          accumulated_cost: 2,
          max_budget_per_task: 10,
          accumulated_token_usage: {
            prompt_tokens: 5,
            completion_tokens: 4,
            cache_read_tokens: 3,
            cache_write_tokens: 2,
            context_window: 100,
            per_turn_token: 9,
          },
        },
        stats,
      });
    });

    it("maps an unknown runtime status to idle", async () => {
      mockGetConversation.mockResolvedValue(
        makeDirectConversation({ execution_status: "future-status" }),
      );

      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-1",
          undefined,
        );

      expect(result.status).toBe("idle");
    });

    it("loads runtime state directly from the conversation runtime when a runtime URL is known", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      // getRuntimeConversation reads the per-conversation runtime agent-server
      // directly at the supplied conversationUrl/sessionApiKey, so the
      // ConversationClient is constructed against those runtime coordinates
      // rather than routed through the cloud proxy.
      mockGetConversation.mockResolvedValue(
        makeDirectConversation({ execution_status: "paused" }),
      );

      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-cloud",
          "http://runtime.example/api/conversations/conv-cloud",
          "session-key",
        );

      expect(result.status).toBe("paused");
      expect(mockGetConversation).toHaveBeenCalledWith("conv-cloud");
      expect(ConversationClient).toHaveBeenLastCalledWith({
        host: "http://runtime.example",
        apiKey: "session-key",
        workingDir: "/workspace/project/agent-canvas",
      });
    });

    it("rejects a cloud runtime read when no runtime coordinates are available", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const proxyRequests = captureRequests(["post"], {});

      await expect(
        AgentServerConversationService.getRuntimeConversation(
          "conv-cloud",
          null,
        ),
      ).rejects.toThrow("No backend is configured");
      expect(mockGetConversation).not.toHaveBeenCalled();
      expect(proxyRequests).toHaveLength(0);
    });

    it("returns an empty hooks result with and without a conversation id", async () => {
      await expect(
        AgentServerConversationService.getHooks(""),
      ).resolves.toEqual({ hooks: [] });
      await expect(
        AgentServerConversationService.getHooks("conv-1"),
      ).resolves.toEqual({ hooks: [] });
    });

    it("normalizes dot segments while keeping a requested file inside the workspace", async () => {
      const encoded = new TextEncoder().encode("plan").buffer;
      mockHttpGet.mockImplementation((url: string) =>
        Promise.resolve({
          data:
            url === "/api/conversations" ? [makeDirectConversation()] : encoded,
        }),
      );

      await expect(
        AgentServerConversationService.readConversationFile(
          "conv-1",
          "/workspace/project/agent-canvas/./notes/../PLAN.md",
        ),
      ).resolves.toBe("plan");
      expect(mockHttpGet).toHaveBeenLastCalledWith(
        "/api/file/download",
        expect.objectContaining({
          params: { path: "/workspace/project/agent-canvas/PLAN.md" },
        }),
      );
    });

    it.each([
      ["relative file path", "PLAN.md", "/workspace/project/agent-canvas"],
      ["root traversal", "/../secrets.txt", "/workspace/project/agent-canvas"],
      ["relative workspace", "/workspace/project/PLAN.md", "workspace/project"],
    ])(
      "rejects an unsafe path caused by a %s",
      async (_label, filePath, workingDir) => {
        mockHttpGet.mockResolvedValue({
          data: [
            makeDirectConversation({ workspace: { working_dir: workingDir } }),
          ],
        });

        await expect(
          AgentServerConversationService.readConversationFile(
            "conv-1",
            filePath,
          ),
        ).rejects.toThrow(
          "Conversation file path must stay inside the workspace",
        );
      },
    );
  });

  describe("conversation lifecycle updates and branching", () => {
    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      setRegisteredBackends([localBackend]);
      setActiveSelection({ backendId: localBackend.id });
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("hydrates repository selections after updating a conversation", async () => {
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ id: "conv-repo" })],
      });

      const first =
        await AgentServerConversationService.updateConversationRepository(
          "conv-repo",
          "OpenHands/agent-canvas",
          "main",
          "github",
        );
      const second =
        await AgentServerConversationService.updateConversationRepository(
          "conv-repo",
          "OpenHands/software-agent-sdk",
          undefined,
          undefined,
        );

      expect(first).toMatchObject({
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "main",
        git_provider: "github",
      });
      expect(second).toMatchObject({
        selected_repository: "OpenHands/software-agent-sdk",
        selected_branch: null,
        git_provider: null,
      });
    });

    it("clears repository selections from a conversation", async () => {
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ id: "conv-repo" })],
      });
      await AgentServerConversationService.updateConversationRepository(
        "conv-repo",
        "OpenHands/agent-canvas",
        "main",
        "github",
      );

      const result =
        await AgentServerConversationService.updateConversationRepository(
          "conv-repo",
          null,
        );

      expect(result).toMatchObject({
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
      });
    });

    it("returns the refreshed conversation after updating its title", async () => {
      mockUpdateConversation.mockResolvedValue(undefined);
      mockHttpGet.mockResolvedValue({
        data: [
          makeDirectConversation({ id: "conv-title", title: "New title" }),
        ],
      });

      const result =
        await AgentServerConversationService.updateConversationTitle(
          "conv-title",
          "New title",
        );

      expect(mockUpdateConversation).toHaveBeenCalledWith("conv-title", {
        title: "New title",
      });
      expect(result.title).toBe("New title");
    });

    it("persists a selected repository when creating a local conversation", async () => {
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
        data: makeDirectConversation({ id: "conv-created" }),
      });

      await AgentServerConversationService.createConversation({
        initialUserMsg: "Inspect the repository",
        metadata: {
          selected_repository: "OpenHands/agent-canvas",
          selected_branch: "main",
          git_provider: "github",
        },
      });
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ id: "conv-created" })],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-created",
        ]);
      expect(conversation).toMatchObject({
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "main",
        git_provider: "github",
        selected_workspace: null,
      });
    });

    it("fails if the local backend disappears while a conversation is being created", async () => {
      mockGetSettings.mockResolvedValue({
        agent_settings: { llm: { model: "gpt-4o" } },
        conversation_settings: {},
      });
      mockGetSettingsForConversation.mockResolvedValue({
        agentSettings: { llm: { model: "gpt-4o" } },
        conversationSettings: {},
        secretsEncrypted: true,
      });
      mockHttpPost.mockImplementation(async () => {
        setRegisteredBackends([]);
        return { data: makeDirectConversation({ id: "conv-created" }) };
      });

      await expect(
        AgentServerConversationService.createConversation(),
      ).rejects.toThrow("No backend is configured");
    });

    it("forwards all supported launch context when creating a cloud conversation", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests = captureRequests(["post"], {
        id: "task-1",
        status: "WORKING",
        request: {},
      });
      const plugins = [
        { source: "github:OpenHands/example-plugin", ref: "v1" },
      ];

      await AgentServerConversationService.createConversation({
        initialUserMsg: "Start here",
        conversationInstructions: "Cloud title",
        plugins,
        metadata: {
          selected_repository: "OpenHands/agent-canvas",
          selected_branch: "main",
          git_provider: "github",
        },
        parentConversationId: "parent-1",
        agentType: "plan",
        sandboxId: "sandbox-1",
        agentProfileId: "profile-1",
      });

      expect(requests).toHaveLength(1);
      expect(requests[0].body).toMatchObject({
        initial_message: {
          role: "user",
          content: [{ type: "text", text: "Start here" }],
        },
        title: "Cloud title",
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "main",
        git_provider: "github",
        plugins,
        parent_conversation_id: "parent-1",
        agent_type: "plan",
        sandbox_id: "sandbox-1",
        agent_profile_id: "profile-1",
      });
    });

    it("uses null cloud launch context when optional values are omitted", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests = captureRequests(["post"], {
        id: "task-1",
        status: "WORKING",
        request: {},
      });

      await AgentServerConversationService.createConversation();

      expect(requests).toHaveLength(1);
      expect(requests[0].body).toMatchObject({
        initial_message: null,
        parent_conversation_id: null,
        sandbox_id: null,
      });
    });

    it("downloads and deletes a conversation through the cloud backend", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests: RecordedRequest[] = [];
      server.use(
        http.get(
          "https://app.all-hands.dev/api/v1/app-conversations/:id/download",
          async ({ request }) => {
            requests.push({
              method: request.method,
              url: request.url,
              headers: Object.fromEntries(request.headers.entries()),
              body: await readJsonBody(request),
            });
            return new HttpResponse("archive");
          },
        ),
        http.delete(
          "https://app.all-hands.dev/api/v1/app-conversations/:id",
          async ({ request }) => {
            requests.push({
              method: request.method,
              url: request.url,
              headers: Object.fromEntries(request.headers.entries()),
              body: await readJsonBody(request),
            });
            return HttpResponse.json({ success: true });
          },
        ),
      );

      const archive =
        await AgentServerConversationService.downloadConversation("conv-cloud");
      expect(await archive.text()).toBe("archive");
      await expect(
        AgentServerConversationService.deleteConversation("conv-cloud"),
      ).resolves.toBeUndefined();

      expect(requests).toHaveLength(2);
      expect(requests[0].method).toBe("GET");
      expect(requests[0].url).toContain("/conv-cloud/download");
      expect(requests[1].method).toBe("DELETE");
      expect(requests[1].url).toContain("/conv-cloud");
    });

    it("updates public sharing through the cloud backend", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const updated = { id: "conv-cloud", public: true };
      const requests = captureRequests(["patch"], updated);

      await expect(
        AgentServerConversationService.updateConversationPublicFlag(
          "conv-cloud",
          true,
        ),
      ).resolves.toEqual(updated);
      expect(requests).toHaveLength(1);
      expect(requests[0].method).toBe("PATCH");
      expect(requests[0].url).toContain("/api/v1/app-conversations/conv-cloud");
      expect(requests[0].body).toEqual({ public: true });
    });

    it("forks at an event, keeps the supplied title, and carries repository metadata", async () => {
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ id: "source-conv" })],
      });
      await AgentServerConversationService.updateConversationRepository(
        "source-conv",
        "OpenHands/agent-canvas",
        "main",
        "github",
      );
      const fork = makeDirectConversation({ id: "fork-conv" });
      mockForkConversation.mockResolvedValue(fork);

      await expect(
        AgentServerConversationService.forkConversation(
          "source-conv",
          "event-7",
          "Investigation branch",
        ),
      ).resolves.toEqual(fork);
      expect(mockForkConversation).toHaveBeenCalledWith("source-conv", {
        from_event_id: "event-7",
        title: "Investigation branch",
      });

      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ id: "fork-conv" })],
      });
      const [forkConversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "fork-conv",
        ]);
      expect(forkConversation).toMatchObject({
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "main",
      });
    });

    it("forks without optional title or source metadata", async () => {
      const fork = makeDirectConversation({ id: "fork-conv" });
      mockForkConversation.mockResolvedValue(fork);

      await AgentServerConversationService.forkConversation(
        "source-without-metadata",
        "event-root",
      );

      expect(mockForkConversation).toHaveBeenCalledWith(
        "source-without-metadata",
        { from_event_id: "event-root" },
      );
    });

    it("rejects conversation branching on the cloud backend", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      await expect(
        AgentServerConversationService.forkConversation(
          "conv-cloud",
          "event-1",
        ),
      ).rejects.toThrow("isn't supported on the cloud backend");
    });

    it("returns an event parent id and treats a root event as parentless", async () => {
      mockGetEvent
        .mockResolvedValueOnce({ parent_id: "parent-event" })
        .mockResolvedValueOnce({ parent_id: null });

      await expect(
        AgentServerConversationService.getEventParentId("conv-1", "event-2"),
      ).resolves.toBe("parent-event");
      await expect(
        AgentServerConversationService.getEventParentId("conv-1", "event-1"),
      ).resolves.toBeUndefined();
    });

    it("activates a cloud profile for the next conversation", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests = captureRequests(["post"], {
        name: "haiku",
        message: "Activated",
        model: "haiku",
      });

      await AgentServerConversationService.switchProfile(null, "haiku");

      expect(requests).toHaveLength(1);
      expect(requests[0].method).toBe("POST");
      expect(requests[0].url).toContain(
        "/api/v1/settings/profiles/haiku/activate",
      );
    });

    it("rejects a local profile that has no usable model", async () => {
      mockGetProfile.mockResolvedValue({
        name: "missing-model",
        config: { model: null },
        api_key_set: false,
      });

      await expect(
        AgentServerConversationService.switchProfile("conv-1", "missing-model"),
      ).rejects.toThrow("Profile 'missing-model' has no model");
      expect(mockSwitchLLM).not.toHaveBeenCalled();
    });

    it("switches an ACP model directly on a local conversation", async () => {
      mockSwitchAcpModel.mockResolvedValue(undefined);

      await AgentServerConversationService.switchAcpModel(
        "conv-1",
        "claude-sonnet-4-6",
      );

      expect(mockSwitchAcpModel).toHaveBeenCalledWith(
        "conv-1",
        "claude-sonnet-4-6",
      );
    });

    it("switches an ACP model through the cloud app endpoint", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests = captureRequests(["post"], { success: true });

      await AgentServerConversationService.switchAcpModel(
        "conv-cloud",
        "claude-opus-4-7",
      );

      expect(requests).toHaveLength(1);
      expect(requests[0].method).toBe("POST");
      expect(requests[0].url).toContain(
        "/api/v1/app-conversations/conv-cloud/switch_acp_model",
      );
      expect(requests[0].body).toEqual({ model: "claude-opus-4-7" });
    });
  });

  describe("mutation-strengthened conversation contracts", () => {
    const metadataStorageKey = "openhands-agent-server-conversation-metadata";
    const invalidResponseMessage =
      "Unable to load conversations because the selected agent server returned " +
      "data this UI does not understand. Check the backend URL/session key and " +
      "update the agent server if needed.";

    const arrangeLocalCreate = (conversationId: string) => {
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
        data: makeDirectConversation({
          id: conversationId,
          created_at: "2026-07-02T10:00:00.000Z",
          updated_at: "2026-07-02T10:01:00.000Z",
        }),
      });
    };

    beforeEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
      setRegisteredBackends([localBackend]);
      setActiveSelection({ backendId: localBackend.id });
    });

    afterEach(() => {
      window.localStorage.clear();
      __resetActiveStoreForTests();
    });

    it("preserves snake-case timestamps and the OpenHands model from the wire response", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          makeDirectConversation({
            created_at: "2026-06-01T09:00:00.000Z",
            updated_at: "2026-06-01T09:15:00.000Z",
            agent: {
              kind: "Agent",
              llm: { model: "openhands/custom-model" },
            },
          }),
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-1",
        ]);

      expect(conversation).toMatchObject({
        created_at: "2026-06-01T09:00:00.000Z",
        updated_at: "2026-06-01T09:15:00.000Z",
        llm_model: "openhands/custom-model",
      });
    });

    it("uses the camel-case updated timestamp when the snake-case field is absent", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          makeDirectConversation({
            updated_at: undefined,
            updatedAt: "2026-06-01T09:30:00.000Z",
          }),
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-1",
        ]);

      expect(conversation?.updated_at).toBe("2026-06-01T09:30:00.000Z");
    });

    it("does not surface a malformed non-string ACP server tag", async () => {
      mockHttpGet.mockResolvedValue({
        data: [
          makeDirectConversation({
            agent: { kind: "ACPAgent", llm: { model: "acp-managed" } },
            tags: { acpserver: 42 },
          }),
        ],
      });

      const [conversation] =
        await AgentServerConversationService.batchGetAppConversations([
          "conv-1",
        ]);

      expect(conversation?.agent_kind).toBe("acp");
      expect(conversation?.acp_server).toBeNull();
    });

    it("returns the complete validation error for a null search response", async () => {
      mockSearchConversations.mockResolvedValue(null);

      await expect(
        AgentServerConversationService.searchConversations(),
      ).rejects.toThrow(invalidResponseMessage);
    });

    it("drops a non-string pagination cursor", async () => {
      mockSearchConversations.mockResolvedValue({
        items: [],
        next_page_id: 17,
      });

      await expect(
        AgentServerConversationService.searchConversations(),
      ).resolves.toEqual({ items: [], next_page_id: null });
    });

    it.each([
      "idle",
      "running",
      "paused",
      "waiting_for_confirmation",
      "finished",
      "error",
      "stuck",
    ] as const)("preserves the supported runtime status %s", async (status) => {
      mockGetConversation.mockResolvedValue(
        makeDirectConversation({ execution_status: status }),
      );

      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-1",
          undefined,
        );

      expect(result.status).toBe(status);
    });

    it("uses a fallback title and the supplied runtime coordinates for a local runtime read", async () => {
      mockGetConversation.mockResolvedValue(
        makeDirectConversation({ title: null }),
      );

      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-1",
          "http://runtime.internal:9000/api/conversations/conv-1",
          "runtime-key",
        );

      expect(result.title).toBe("Conversation conv-");
      expect(ConversationClient).toHaveBeenLastCalledWith({
        host: "http://runtime.internal:9000",
        apiKey: "runtime-key",
        workingDir: "/workspace/project/agent-canvas",
      });
    });

    it("constructs the VS Code client with the supplied runtime coordinates", async () => {
      mockHttpGet.mockResolvedValue({ data: [makeDirectConversation()] });
      mockVSCodeGetUrl.mockResolvedValue("http://localhost:3000/vscode");

      await AgentServerConversationService.getVSCodeUrl(
        "conv-1",
        "http://runtime.internal:9000/api/conversations/conv-1",
        "runtime-key",
      );

      expect(VSCodeClient).toHaveBeenCalledWith({
        host: "http://runtime.internal:9000",
        apiKey: "runtime-key",
        workingDir: "/workspace/project/agent-canvas",
      });
    });

    it.each([
      [
        "session key",
        {
          conversationUrl:
            "http://caller-runtime.example/api/conversations/conv-cloud",
          sessionApiKey: null,
        },
      ],
      [
        "conversation URL",
        { conversationUrl: null, sessionApiKey: "caller-key" },
      ],
    ] as const)(
      "refreshes cloud runtime credentials when only the %s is missing",
      async (_missingField, runtime) => {
        setRegisteredBackends([cloudBackend]);
        setActiveSelection({ backendId: cloudBackend.id });
        const requests: RecordedRequest[] = [];
        server.use(
          http.get(
            "https://app.all-hands.dev/api/v1/app-conversations",
            async ({ request }) => {
              requests.push({
                method: request.method,
                url: request.url,
                headers: Object.fromEntries(request.headers.entries()),
                body: await readJsonBody(request),
              });
              return HttpResponse.json([
                {
                  id: "conv-cloud",
                  conversation_url:
                    "http://fetched-runtime.example/api/conversations/conv-cloud",
                  session_api_key: "fetched-key",
                },
              ]);
            },
          ),
          http.post("*/api/cloud-proxy", async ({ request }) => {
            requests.push({
              method: request.method,
              url: request.url,
              headers: Object.fromEntries(request.headers.entries()),
              body: await readJsonBody(request),
            });
            return HttpResponse.json({});
          }),
        );

        await AgentServerConversationService.sendMessage(
          "conv-cloud",
          message,
          runtime,
        );

        const batchGets = requests.filter((req) => req.method === "GET");
        const proxyPost = requests.find(
          (req) => req.url === "http://localhost:54928/api/cloud-proxy",
        );
        expect(batchGets).toHaveLength(1);
        expect(proxyPost?.body).toMatchObject({
          host: "http://fetched-runtime.example",
          headers: { "X-Session-API-Key": "fetched-key" },
        });
      },
    );

    it.each([
      ["conversation URL", { session_api_key: "fetched-key" }],
      [
        "session key",
        {
          conversation_url:
            "http://runtime.example/api/conversations/conv-cloud",
        },
      ],
    ])(
      "returns the startup error when the fetched cloud conversation lacks its %s",
      async (_missingField, cloudConversation) => {
        setRegisteredBackends([cloudBackend]);
        setActiveSelection({ backendId: cloudBackend.id });
        const proxyRequests = captureRequests(["post"], {});
        server.use(
          http.get(
            "https://app.all-hands.dev/api/v1/app-conversations",
            () =>
              HttpResponse.json([{ id: "conv-cloud", ...cloudConversation }]),
          ),
        );

        await expect(
          AgentServerConversationService.sendMessage("conv-cloud", message),
        ).rejects.toThrow(
          "Conversation sandbox is still starting. Wait for it to finish, then try again.",
        );
        expect(proxyRequests).toHaveLength(0);
      },
    );

    it("authenticates a cloud runtime state read with its session key", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      // getRuntimeConversation reads the conversation runtime directly through
      // ConversationClient built from the supplied runtime coordinates, so the
      // session key travels as the client apiKey rather than via the proxy.
      mockGetConversation.mockResolvedValue(
        makeDirectConversation({ execution_status: "running" }),
      );

      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-cloud",
          "http://runtime.example/api/conversations/conv-cloud",
          "runtime-key",
        );

      expect(result.status).toBe("running");
      expect(mockGetConversation).toHaveBeenCalledWith("conv-cloud");
      expect(ConversationClient).toHaveBeenLastCalledWith({
        host: "http://runtime.example",
        apiKey: "runtime-key",
        workingDir: "/workspace/project/agent-canvas",
      });
    });

    it("rejects traversal above root even when the remaining cloud path points into the workspace", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      const requests = captureRequests(["get"], "");

      await expect(
        AgentServerConversationService.readConversationFile(
          "conv-cloud",
          "/../workspace/project/PLAN.md",
        ),
      ).rejects.toThrow(
        "Conversation file path must stay inside the workspace",
      );
      expect(requests).toHaveLength(0);
    });

    it("allows an explicit path equal to the local workspace root", async () => {
      const encoded = new TextEncoder().encode("workspace root").buffer;
      mockHttpGet.mockImplementation((url: string) =>
        Promise.resolve({
          data:
            url === "/api/conversations" ? [makeDirectConversation()] : encoded,
        }),
      );

      await expect(
        AgentServerConversationService.readConversationFile(
          "conv-1",
          "/workspace/project/agent-canvas",
        ),
      ).resolves.toBe("workspace root");
      expect(mockHttpGet).toHaveBeenLastCalledWith(
        "/api/file/download",
        expect.objectContaining({
          params: { path: "/workspace/project/agent-canvas" },
        }),
      );
    });

    it("returns the complete local start task and persists an attached workspace", async () => {
      arrangeLocalCreate("conv-created");
      const plugins = [
        { source: "github:OpenHands/example-plugin", ref: "v1" },
      ];

      const result = await AgentServerConversationService.createConversation({
        initialUserMsg: "Inspect the repository",
        plugins,
        workingDirOverride: "/Users/jane/projects/canvas",
      });

      expect(result).toEqual({
        id: "conv-created",
        created_by_user_id: null,
        status: "READY",
        detail: null,
        app_conversation_id: "conv-created",
        agent_server_url: "http://localhost:54928",
        request: {
          initial_message: {
            role: "user",
            content: [{ type: "text", text: "Inspect the repository" }],
            run: true,
          },
          plugins,
        },
        created_at: "2026-07-02T10:00:00.000Z",
        updated_at: "2026-07-02T10:01:00.000Z",
      });
      expect(getStoredConversationMetadata("conv-created")).toEqual({
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        selected_workspace: "/Users/jane/projects/canvas",
        workspace_mode: "local_repo",
      });
    });

    it("does not persist empty metadata for a conversation without a selected source", async () => {
      arrangeLocalCreate("conv-unselected");

      await AgentServerConversationService.createConversation();

      expect(getStoredConversationMetadata("conv-unselected")).toBeNull();
      expect(window.localStorage.getItem(metadataStorageKey)).toBeNull();
    });

    it("preserves attached-workspace metadata while selecting a repository", async () => {
      setStoredConversationMetadata("conv-repo", {
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        selected_workspace: "/Users/jane/projects/canvas",
        workspace_mode: "local_repo",
      });
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ id: "conv-repo" })],
      });

      const result =
        await AgentServerConversationService.updateConversationRepository(
          "conv-repo",
          "OpenHands/agent-canvas",
          "main",
          "github",
        );

      expect(result.selected_workspace).toBe("/Users/jane/projects/canvas");
      expect(getStoredConversationMetadata("conv-repo")).toEqual({
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "main",
        git_provider: "github",
        selected_workspace: "/Users/jane/projects/canvas",
        workspace_mode: "local_repo",
      });
    });

    it("removes all source metadata when the selected repository is cleared", async () => {
      setStoredConversationMetadata("conv-repo", {
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "main",
        git_provider: "github",
        selected_workspace: "/Users/jane/projects/canvas",
        workspace_mode: "local_repo",
      });
      mockHttpGet.mockResolvedValue({
        data: [makeDirectConversation({ id: "conv-repo" })],
      });

      const result =
        await AgentServerConversationService.updateConversationRepository(
          "conv-repo",
          null,
        );

      expect(result.selected_repository).toBeNull();
      expect(result.selected_workspace).toBeNull();
      expect(getStoredConversationMetadata("conv-repo")).toBeNull();
    });

    it("does not persist an empty metadata record when forking a conversation without source metadata", async () => {
      mockForkConversation.mockResolvedValue(
        makeDirectConversation({ id: "fork-without-metadata" }),
      );

      await AgentServerConversationService.forkConversation(
        "source-without-metadata",
        "event-root",
      );

      expect(getStoredConversationMetadata("fork-without-metadata")).toBeNull();
      expect(window.localStorage.getItem(metadataStorageKey)).toBeNull();
    });

    it("rejects a profile whose model is a non-string value", async () => {
      mockGetProfile.mockResolvedValue({
        name: "numeric-model",
        config: { model: 7 },
        api_key_set: false,
      });

      await expect(
        AgentServerConversationService.switchProfile("conv-1", "numeric-model"),
      ).rejects.toThrow("Profile 'numeric-model' has no model");
      expect(mockSwitchLLM).not.toHaveBeenCalled();
    });

    it("checks subscription authentication against the selected profile config", async () => {
      const status = vi
        .spyOn(LLMSubscriptionService, "getOpenAIStatus")
        .mockResolvedValue({
          vendor: OPENAI_SUBSCRIPTION_VENDOR,
          connected: false,
          accountEmail: null,
          expiresAt: null,
        });
      mockGetProfile.mockResolvedValue({
        name: "subscription-profile",
        config: {
          model: "openai/gpt-5.1-codex",
          auth_type: LLM_AUTH_TYPE_SUBSCRIPTION,
          subscription_vendor: OPENAI_SUBSCRIPTION_VENDOR,
        },
        api_key_set: false,
      });

      try {
        await expect(
          AgentServerConversationService.switchProfile(
            "conv-1",
            "subscription-profile",
          ),
        ).rejects.toThrow(
          "Connect your ChatGPT subscription before starting a conversation with this LLM profile.",
        );
        expect(status).toHaveBeenCalledOnce();
        expect(mockSwitchLLM).not.toHaveBeenCalled();
      } finally {
        status.mockRestore();
      }
    });
  });
});
