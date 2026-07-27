import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { SuggestedTask } from "#/utils/types";
import {
  getStoredConversationMetadata,
  removeStoredConversationMetadata,
} from "#/api/conversation-metadata-store";

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
  }),
}));

// The default→agent_settings downgrade is local-only (#1571 review); default
// to local so the existing (pre-review) assertions below are unaffected, and
// override per-test to exercise the cloud path.
interface MockActiveBackend {
  backend: { id: string; kind: "local" | "cloud" };
  orgId: string | null;
}
const mockUseActiveBackend = vi.fn<() => MockActiveBackend>(() => ({
  backend: { id: "local-1", kind: "local" },
  orgId: null,
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => mockUseActiveBackend(),
}));

// The hook stamps the active LLM profile onto the conversation (#1082).
// Mock it so the captured value is deterministic — the real hook fires a
// query the global MSW layer would answer non-deterministically under test
// timing.
const { useLlmProfilesMock } = vi.hoisted(() => ({
  useLlmProfilesMock: vi.fn(() => ({
    data: { active_profile: null as string | null },
  })),
}));
vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => useLlmProfilesMock(),
}));

// The hook warms the agent-profiles cache; the launch path itself awaits the
// list via ensureQueryData (so it can't race a cold cache). Mock the hook to a
// no-op and drive launches through the service mock below.
vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => ({ data: undefined }),
}));

// The launch path resolves the active AgentProfile by awaiting
// `AgentProfilesService.listProfiles` through the query cache (#3727).
// Default: no active profile, so a plain create stays on the legacy path.
const { listAgentProfilesMock } = vi.hoisted(() => ({
  listAgentProfilesMock: vi.fn(),
}));
vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: listAgentProfilesMock },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));
listAgentProfilesMock.mockResolvedValue({
  profiles: [],
  active_agent_profile_id: null,
});

// LLM-profile service: real listProfiles calls (the llmProfileExists
// validation — see use-create-conversation.ts) so it can be asserted per-test.
const listLlmProfilesMock = vi.fn();
vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  __esModule: true,
  default: {
    listProfiles: (...args: unknown[]) => listLlmProfilesMock(...args),
  },
}));
listLlmProfilesMock.mockResolvedValue({ profiles: [], active_profile: null });

describe("useCreateConversation", () => {
  afterEach(() => {
    // Restore the default (no active AgentProfile) so the overrides below
    // don't leak into the other create-call assertions.
    mockUseActiveBackend.mockReturnValue({
      backend: { id: "local-1", kind: "local" as const },
      orgId: null,
    });
    listAgentProfilesMock.mockReset();
    listAgentProfilesMock.mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    });
    listLlmProfilesMock.mockReset();
    listLlmProfilesMock.mockResolvedValue({
      profiles: [],
      active_profile: null,
    });
    useLlmProfilesMock.mockReturnValue({ data: { active_profile: null } });
    removeStoredConversationMetadata("conv-with-plugins");
    removeStoredConversationMetadata("conv-ref-stamp");
  });

  it("passes suggested tasks to the V1 create conversation API", async () => {
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        created_by_user_id: null,
        status: "READY",
        detail: null,
        app_conversation_id: null,
        agent_server_url: "http://agent-server.local",
        request: {
          initial_message: {
            role: "user",
            content: [{ type: "text", text: "Please address the comments" }],
          },
          processors: [],
          llm_model: null,
          selected_repository: null,
          selected_branch: null,
          git_provider: "github",
          suggested_task: null,
          title: null,
          trigger: null,
          pr_number: [],
          parent_conversation_id: null,
          agent_type: "default",
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    const suggestedTask: SuggestedTask = {
      git_provider: "github",
      issue_number: 42,
      repo: "owner/repo",
      title: "Resolve comments",
      task_type: "UNRESOLVED_COMMENTS",
    };

    await result.current.mutateAsync({
      query: "Please address the comments",
      repository: {
        name: "owner/repo",
        gitProvider: "github",
        branch: "main",
      },
      conversationInstructions: "Focus on review comments",
      suggestedTask,
    });

    await waitFor(() => {
      expect(createConversationSpy).toHaveBeenCalledWith(
        "Please address the comments",
        "Focus on review comments",
        undefined,
        {
          selected_repository: "owner/repo",
          selected_branch: "main",
          git_provider: "github",
        },
        undefined,
        undefined,
        undefined,
        undefined,
      );
    });
  });

  it("launches new local conversations from the active AgentProfile (#3727)", async () => {
    listAgentProfilesMock.mockResolvedValue({
      profiles: [],
      active_agent_profile_id: "profile-abc",
    });
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        app_conversation_id: "conv-1",
        agent_server_url: "http://agent-server.local",
      } as never);

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    await waitFor(() => {
      // sandboxId (#9) is undefined; the active profile id rides as #10.
      const call = createConversationSpy.mock.lastCall;
      expect(call?.[8]).toBeUndefined();
      expect(call?.[9]).toBe("profile-abc");
    });
  });

  it("awaits the profiles fetch so an early send still launches from the active profile", async () => {
    // A send fired before the home profiles query resolves must block on the
    // fetch, not fall through to the agent_settings path (#1571 review F2).
    let resolveList: (value: unknown) => void = () => {};
    listAgentProfilesMock.mockReturnValue(
      new Promise((resolve) => {
        resolveList = resolve;
      }),
    );
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        app_conversation_id: "conv-1",
        agent_server_url: "http://agent-server.local",
      } as never);
    // Spies persist across tests in this file; drop earlier calls so the
    // not-yet-called assertion below sees only this launch.
    createConversationSpy.mockClear();

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    const pending = result.current.mutateAsync({ query: "hello" });
    expect(createConversationSpy).not.toHaveBeenCalled();

    resolveList({ profiles: [], active_agent_profile_id: "profile-late" });
    await pending;

    const call = createConversationSpy.mock.lastCall;
    expect(call?.[9]).toBe("profile-late");
  });

  it("falls back to the agent_settings launch when the profiles fetch fails", async () => {
    listAgentProfilesMock.mockRejectedValue(new Error("not supported"));
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        app_conversation_id: "conv-1",
        agent_server_url: "http://agent-server.local",
      } as never);

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    // Resolves without stalling: the launch-path fetch is retry: false.
    await result.current.mutateAsync({ query: "hello" });

    // No profile tail — the create stays on the legacy agent_settings path.
    const call = createConversationSpy.mock.lastCall;
    expect(call?.[9]).toBeUndefined();
  });

  it("invalidates the conversation list and start-tasks queries on success", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockResolvedValue({
      id: "task-id",
      created_by_user_id: null,
      status: "READY",
      detail: null,
      app_conversation_id: "conv-1",
      agent_server_url: "http://agent-server.local",
      request: {
        initial_message: null,
        processors: [],
        llm_model: null,
        selected_repository: null,
        selected_branch: null,
        git_provider: "github",
        suggested_task: null,
        title: null,
        trigger: null,
        pr_number: [],
        parent_conversation_id: null,
        agent_type: "default",
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    const queryClient = new QueryClient();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({});

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversations"],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["start-tasks"],
      });
    });
  });

  it("persists explicitly-attached plugins (coordinates only) to conversation metadata at creation", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockResolvedValue({
      id: "task-id",
      created_by_user_id: null,
      status: "READY",
      detail: null,
      app_conversation_id: "conv-with-plugins",
      agent_server_url: "http://agent-server.local",
      request: {
        initial_message: null,
        processors: [],
        llm_model: null,
        selected_repository: null,
        selected_branch: null,
        git_provider: "github",
        suggested_task: null,
        title: null,
        trigger: null,
        pr_number: [],
        parent_conversation_id: null,
        agent_type: "default",
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({
      plugins: [
        {
          source: "github:o/a",
          ref: null,
          repo_path: "plugins/a",
          parameters: { token: "secret" },
        },
      ],
    });

    await waitFor(() =>
      expect(
        getStoredConversationMetadata("conv-with-plugins")?.plugins,
      ).toEqual([{ source: "github:o/a", ref: null, repo_path: "plugins/a" }]),
    );
  });

  it("launches the seeded `default` profile via agent_settings so canvas enrichments survive", async () => {
    // The active profile IS the well-known default → it's the enriched baseline
    // (mirrors agent_settings), not a deliberate profile pick, so the launch
    // stays on the agent_settings path (no profile tail) even though its
    // llm_profile_ref resolves. Keeps <RUNTIME_SERVICES> and project skills,
    // which the profile-resolution path drops.
    listAgentProfilesMock.mockResolvedValue({
      profiles: [
        {
          id: "profile-default",
          name: "default",
          agent_kind: "openhands",
          revision: 1,
          llm_profile_ref: "gpt",
          mcp_server_refs: null,
        },
      ],
      active_agent_profile_id: "profile-default",
    });
    listLlmProfilesMock.mockResolvedValue({
      profiles: [{ name: "gpt" }],
      active_profile: "gpt",
    });
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        app_conversation_id: "conv-1",
        agent_server_url: "http://agent-server.local",
      } as never);

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    const call = createConversationSpy.mock.lastCall;
    expect(call?.[9]).toBeUndefined();
  });

  it("keeps the profile path for an ACP `default` profile (agent_settings can't carry ACP config)", async () => {
    // The default→agent_settings shortcut is OpenHands-only: activation is
    // pointer-only, so global agent_settings is stale (still OpenHands) for an
    // active ACP profile — routing it via agent_settings would launch the wrong
    // agent. An ACP `default` must resolve server-side via agent_profile_id.
    listAgentProfilesMock.mockResolvedValue({
      profiles: [
        {
          id: "profile-acp-default",
          name: "default",
          agent_kind: "acp",
          revision: 1,
          llm_profile_ref: null,
          mcp_server_refs: null,
        },
      ],
      active_agent_profile_id: "profile-acp-default",
    });
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        app_conversation_id: "conv-1",
        agent_server_url: "http://agent-server.local",
      } as never);

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    const call = createConversationSpy.mock.lastCall;
    expect(call?.[9]).toBe("profile-acp-default");
    expect(call?.[10]).toBe("acp");
  });

  it("launches the seeded `default` profile from its resolved id on cloud (no agent_settings fallback exists there) (#1571)", async () => {
    // The local-only downgrade exists to preserve canvas-only enrichments that
    // only ride the agent_settings path; cloud has no such payload, so the
    // seeded `default` must always launch via agent_profile_id there — a
    // downgrade there would send agent_profile_id: null and the conversation
    // would never get `launched_agent_profile` stamped.
    mockUseActiveBackend.mockReturnValue({
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    });
    listAgentProfilesMock.mockResolvedValue({
      profiles: [
        {
          id: "profile-default",
          name: "default",
          agent_kind: "openhands",
          revision: 1,
          llm_profile_ref: "gpt",
          mcp_server_refs: null,
        },
      ],
      active_agent_profile_id: "profile-default",
    });
    listLlmProfilesMock.mockResolvedValue({
      profiles: [{ name: "gpt" }],
      active_profile: "gpt",
    });
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        app_conversation_id: "conv-1",
        agent_server_url: "http://agent-server.local",
      } as never);

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    const call = createConversationSpy.mock.lastCall;
    expect(call?.[9]).toBe("profile-default");
  });

  it("stamps the launched openhands profile's llm_profile_ref into conversation metadata (#1082)", async () => {
    // A named (non-default) profile launches via the profile path and runs its
    // own llm_profile_ref — which differs from the standalone active LLM
    // profile — so the switcher pill must name the ref, not the active profile.
    useLlmProfilesMock.mockReturnValue({
      data: { active_profile: "standalone-active" },
    });
    listAgentProfilesMock.mockResolvedValue({
      profiles: [
        {
          id: "profile-custom",
          name: "My Profile",
          agent_kind: "openhands",
          revision: 2,
          llm_profile_ref: "claude",
          mcp_server_refs: null,
        },
      ],
      active_agent_profile_id: "profile-custom",
    });
    listLlmProfilesMock.mockResolvedValue({
      profiles: [{ name: "claude" }],
      active_profile: "standalone-active",
    });
    const createConversationSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        app_conversation_id: "conv-ref-stamp",
        agent_server_url: "http://agent-server.local",
      } as never);

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    const call = createConversationSpy.mock.lastCall;
    expect(call?.[9]).toBe("profile-custom");
    await waitFor(() =>
      expect(
        getStoredConversationMetadata("conv-ref-stamp")?.active_profile,
      ).toBe("claude"),
    );
  });
});
