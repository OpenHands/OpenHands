import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useForkConversation } from "#/hooks/mutation/use-fork-conversation";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getConversationState } from "#/utils/conversation-local-storage";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const { navigateMock } = vi.hoisted(() => ({ navigateMock: vi.fn() }));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      getEventParentId: vi.fn(),
      forkConversation: vi.fn(),
    },
  }),
);

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: navigateMock }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

const createWrapper = (client?: QueryClient) => {
  const queryClient =
    client ??
    new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { Wrapper, queryClient };
};

describe("useForkConversation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("navigates to the fork and seeds its composer draft when the message is excluded", async () => {
    vi.mocked(
      AgentServerConversationService.getEventParentId,
    ).mockResolvedValue("evt-parent");
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockResolvedValue({
      id: "fork-1",
      leaf_event_id: "evt-parent",
    } as never);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      sourceConversationId: "conv-1",
      eventId: "evt-2",
      editText: "original text",
    });

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1"),
    );
    expect(
      AgentServerConversationService.forkConversation,
    ).toHaveBeenCalledWith("conv-1", "evt-parent", undefined);
    expect(getConversationState("fork-1").draftMessage).toBe("original text");
  });

  it("does not prefill when an older agent-server ignores the branch point", async () => {
    vi.mocked(
      AgentServerConversationService.getEventParentId,
    ).mockResolvedValue("evt-parent");
    // HEAD is not the requested branch point: the whole conversation was
    // copied, so the message was not excluded and prefilling would duplicate it.
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockResolvedValue({
      id: "fork-1",
      leaf_event_id: "evt-2",
    } as never);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      sourceConversationId: "conv-1",
      eventId: "evt-2",
      editText: "original text",
    });

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1"),
    );
    expect(getConversationState("fork-1").draftMessage).toBeNull();
  });

  it("completes the fork even after the invoking row unmounts", async () => {
    // A virtualized chat can scroll the source row out of the window while the
    // fork request is in flight. The completion side effects live on the
    // mutation, so they must still run.
    let resolveFork: (value: unknown) => void = () => {};
    vi.mocked(AgentServerConversationService.forkConversation).mockReturnValue(
      new Promise((resolve) => {
        resolveFork = resolve;
      }) as never,
    );

    const { Wrapper } = createWrapper();
    const { result, unmount } = renderHook(() => useForkConversation(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      sourceConversationId: "conv-1",
      eventId: "evt-2",
    });
    unmount();

    resolveFork({ id: "fork-1" });

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1"),
    );
  });

  it("surfaces a failed fork as an error toast", async () => {
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockRejectedValue(new Error("fork failed"));

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation(), {
      wrapper: Wrapper,
    });

    result.current.mutate({ sourceConversationId: "conv-1", eventId: "evt-2" });

    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith("fork failed"),
    );
    expect(navigateMock).not.toHaveBeenCalled();
  });
});
