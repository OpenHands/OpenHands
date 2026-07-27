/**
 * Regression test for the automation-detail-page bug where every
 * activity-log row mounted a (closed) RunLogsModal, each of which fired a
 * `/api/conversations?ids=<id>` request on page load.
 *
 * useBashCommandLogs must honor its `enabled` flag (set to the modal's
 * `isOpen`) when resolving the conversation: a closed modal fetches
 * nothing; opening it resolves exactly one conversation lookup.
 *
 * Per the testing rules we exercise the REAL useUserConversation and mock
 * only the underlying service it depends on
 * (AgentServerConversationService.batchGetAppConversations).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useBashCommandLogs } from "#/hooks/query/use-bash-command-logs";

// ── Mocks ─────────────────────────────────────────────────────────────────────

const { batchGetAppConversationsMock, listOutputsMock, useActiveBackendMock } =
  vi.hoisted(() => ({
    batchGetAppConversationsMock: vi.fn(),
    listOutputsMock: vi.fn(),
    useActiveBackendMock: vi.fn(),
  }));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      batchGetAppConversations: (...args: unknown[]) =>
        batchGetAppConversationsMock(...args),
    },
  }),
);

vi.mock("#/api/bash-service/bash-service.api", () => ({
  default: { listOutputs: (...args: unknown[]) => listOutputsMock(...args) },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build a wrapper whose QueryClient is created once and stays stable for the
// lifetime of the test. Creating the client inside the component body would
// re-instantiate it on every wrapper re-render (e.g. a `rerender` call),
// discarding the cache and tripping TanStack Query dev-mode warnings.
function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

beforeEach(() => {
  batchGetAppConversationsMock.mockReset();
  listOutputsMock.mockReset();
  useActiveBackendMock.mockReset();

  useActiveBackendMock.mockReturnValue({
    backend: { id: "bk-1", kind: "local" },
    orgId: null,
  });
  listOutputsMock.mockResolvedValue([]);
  batchGetAppConversationsMock.mockResolvedValue([
    { conversation_url: null, session_api_key: null, sandbox_status: null },
  ]);
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("useBashCommandLogs — gates the conversation lookup on `enabled`", () => {
  it("does not fetch the conversation while the modal is closed (enabled=false)", () => {
    // Arrange + Act: a RunLogsModal mounted closed for an activity-log row.
    renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
          enabled: false,
        }),
      { wrapper: makeWrapper() },
    );

    // Assert: no /api/conversations request is issued on page load.
    expect(batchGetAppConversationsMock).not.toHaveBeenCalled();
  });

  it("fetches the conversation exactly once when the modal opens (enabled=true)", async () => {
    // Arrange + Act: the user opens the row's logs modal.
    renderHook(
      () =>
        useBashCommandLogs({
          conversationId: "conv-1",
          bashCommandId: "cmd-1",
          enabled: true,
        }),
      { wrapper: makeWrapper() },
    );

    // Assert: one lookup fires for that conversation id (functionality preserved).
    await waitFor(() =>
      expect(batchGetAppConversationsMock).toHaveBeenCalledWith(["conv-1"]),
    );
    expect(batchGetAppConversationsMock).toHaveBeenCalledTimes(1);
  });
});
