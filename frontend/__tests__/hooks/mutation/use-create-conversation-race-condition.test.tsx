import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";

// ---------------------------------------------------------------
// These tests document a race condition where clicking
// "New Conversation" before settings have loaded causes
// the hook to create a V0 (legacy) conversation instead of V1.
//
// Root cause (original code):
//   const { data: settings } = useSettings();
//   ...
//   const useV1 = !!settings?.v1_enabled && !createMicroagent;
//
// When settings haven't loaded yet, `settings` is `undefined`,
// so `!!undefined?.v1_enabled` → false, silently routing through
// the V0 code path even though the backend defaults v1_enabled
// to `true`.
//
// The fix uses `queryClient.ensureQueryData()` inside the mutation
// to wait for settings before deciding V0 vs V1, with a fallback
// to DEFAULT_SETTINGS (v1_enabled: true) on fetch failure.
// ---------------------------------------------------------------

const mockGetSettingsQueryFn = vi.fn();

vi.mock("#/hooks/query/use-settings", async () => {
  const actual = await vi.importActual<
    typeof import("#/hooks/query/use-settings")
  >("#/hooks/query/use-settings");
  return {
    ...actual,
    getSettingsQueryFn: (...args: unknown[]) =>
      mockGetSettingsQueryFn(...args),
  };
});

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
  }),
}));

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({ organizationId: null }),
}));

// Shared mock return values
const V1_RESPONSE = {
  id: "task-id-123",
  created_by_user_id: null,
  status: "READY" as const,
  detail: null,
  app_conversation_id: null,
  sandbox_id: null,
  agent_server_url: "http://agent-server.local",
  request: {
    sandbox_id: null,
    initial_message: { role: "user" as const, content: [{ type: "text" as const, text: "hello" }] },
    processors: [],
    llm_model: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: "github" as const,
    suggested_task: null,
    title: null,
    trigger: null,
    pr_number: [],
    parent_conversation_id: null,
    agent_type: "default" as const,
  },
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

const V0_RESPONSE = {
  conversation_id: "conv-legacy",
  session_api_key: null,
  url: null,
  title: "",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  last_updated_at: new Date().toISOString(),
  status: "RUNNING" as const,
  runtime_status: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
};

describe("useCreateConversation – V0 race condition", () => {
  let v1Spy: ReturnType<typeof vi.spyOn>;
  let v0Spy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.clearAllMocks();

    v1Spy = vi
      .spyOn(V1ConversationService, "createConversation")
      .mockResolvedValue(V1_RESPONSE);

    v0Spy = vi
      .spyOn(ConversationService, "createConversation")
      .mockResolvedValue(V0_RESPONSE);
  });

  /**
   * BUG REPRODUCTION: When the settings API hasn't been called yet
   * (no cached data), the hook should wait for settings to load
   * rather than defaulting to V0.
   *
   * The fix uses `ensureQueryData` to fetch/wait for settings before
   * deciding V0 vs V1.  The mock here resolves with v1_enabled: true,
   * proving that the mutation waits for the settings query.
   */
  it("should use V1 API even when settings are not yet cached (race condition scenario)", async () => {
    // Simulate the race condition: settings haven't been fetched yet.
    // With the fix, ensureQueryData will call getSettingsQueryFn to fetch them.
    mockGetSettingsQueryFn.mockResolvedValue({ v1_enabled: true });

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    await waitFor(() => {
      // V1 should be used — the fix waits for settings before deciding
      expect(v1Spy).toHaveBeenCalled();
    });

    // V0 should NOT have been called
    expect(v0Spy).not.toHaveBeenCalled();
  });

  /**
   * When the settings fetch fails (e.g. 404 for a new user), the hook
   * falls back to DEFAULT_SETTINGS where v1_enabled is now `true`,
   * still routing through V1.
   */
  it("should use V1 API when settings fetch fails (falls back to defaults)", async () => {
    // Simulate settings API failure
    mockGetSettingsQueryFn.mockRejectedValue(new Error("404 Not Found"));

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    await waitFor(() => {
      // DEFAULT_SETTINGS.v1_enabled is now true, so V1 should be used
      expect(v1Spy).toHaveBeenCalled();
    });
    expect(v0Spy).not.toHaveBeenCalled();
  });

  /**
   * When settings explicitly have v1_enabled: true, V1 API is used.
   */
  it("should use V1 API when settings explicitly have v1_enabled: true", async () => {
    mockGetSettingsQueryFn.mockResolvedValue({ v1_enabled: true });

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    await waitFor(() => {
      expect(v1Spy).toHaveBeenCalled();
    });
    expect(v0Spy).not.toHaveBeenCalled();
  });

  /**
   * When v1_enabled is explicitly false, V0 should be used.
   */
  it("should use V0 API when v1_enabled is explicitly false", async () => {
    mockGetSettingsQueryFn.mockResolvedValue({ v1_enabled: false });

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

    await result.current.mutateAsync({ query: "hello" });

    await waitFor(() => {
      expect(v0Spy).toHaveBeenCalled();
    });
    expect(v1Spy).not.toHaveBeenCalled();
  });
});
