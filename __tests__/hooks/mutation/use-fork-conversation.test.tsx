import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useForkConversation } from "#/hooks/mutation/use-fork-conversation";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getConversationState } from "#/utils/conversation-local-storage";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToastWithLink,
} from "#/utils/custom-toast-handlers";

const { navigateMock, navigationState } = vi.hoisted(() => ({
  navigateMock: vi.fn(),
  // The route the navigation owner last supplied, plus its optional live
  // getter. Left unset so tests that only set `window.location` keep exercising
  // the document-URL fallback; an embedded host supplies the route (and may
  // supply a getter) without touching the URL.
  navigationState: {
    currentPath: undefined as string | undefined,
    getCurrentPath: undefined as (() => string) | undefined,
  },
}));

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
  useNavigation: () => ({
    navigate: navigateMock,
    currentPath: navigationState.currentPath,
    getCurrentPath: navigationState.getCurrentPath,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
  displaySuccessToastWithLink: vi.fn(),
}));

// The hook falls back to `window.location.pathname` when the navigation owner
// exposes no route, so each test parks the jsdom URL there.
const setLocation = (pathname: string) => {
  window.history.replaceState({}, "", pathname);
};

beforeEach(() => {
  setLocation("/conversations/conv-1");
  navigationState.currentPath = undefined;
  navigationState.getCurrentPath = undefined;
});

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
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({
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
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({
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
    const { result, unmount } = renderHook(
      () => useForkConversation("conv-1"),
      {
        wrapper: Wrapper,
      },
    );

    result.current.mutate({
      eventId: "evt-2",
    });
    unmount();

    resolveFork({ id: "fork-1" });

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1"),
    );
  });

  it("does not hijack a conversation the reader switched to, linking the fork instead", async () => {
    // The mutation callback outlives the row, so the reader may have navigated
    // away while the fork was in flight. Dragging them into the fork would
    // hijack that navigation; they get a toast with a link instead.
    setLocation("/conversations/conv-2");
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockResolvedValue({ id: "fork-1" } as never);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({ eventId: "evt-2" });

    await waitFor(() =>
      expect(displaySuccessToastWithLink).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        "/conversations/fork-1",
        expect.any(Function),
      ),
    );
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("still navigates from a sub-route of the source conversation", async () => {
    // The conversation's panel route is a sub-path of the conversation, so the
    // reader is still "in" the source and the fork should open.
    setLocation("/conversations/conv-1/panel");
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockResolvedValue({ id: "fork-1" } as never);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({ eventId: "evt-2" });

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1"),
    );
    expect(displaySuccessToastWithLink).not.toHaveBeenCalled();
  });

  it("opens the fork through the host when an embedded Canvas supplies the route without changing the document URL", async () => {
    // The host supplies the route: the browser stays on /workspace while Canvas
    // shows conv-1. Reading `window.location` would report the reader as having
    // left the source, so the fork would only be linked.
    setLocation("/workspace");
    navigationState.currentPath = "/conversations/conv-1";
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockResolvedValue({ id: "fork-1" } as never);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({ eventId: "evt-2" });

    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1"),
    );
    expect(displaySuccessToastWithLink).not.toHaveBeenCalled();
  });

  it("reads the live route from the host getter at settle time, not the path captured when the row rendered", async () => {
    // The row rendered while Canvas showed conv-1, but the reader navigated to
    // conv-2 before the fork settled. The getter reflects that; the captured
    // `currentPath` would not, and would wrongly pull the reader into the fork.
    setLocation("/workspace");
    navigationState.currentPath = "/conversations/conv-1";
    let route = "/conversations/conv-1";
    navigationState.getCurrentPath = () => route;
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockResolvedValue({ id: "fork-1" } as never);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({ eventId: "evt-2" });
    route = "/conversations/conv-2";

    await waitFor(() =>
      expect(displaySuccessToastWithLink).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        "/conversations/fork-1",
        expect.any(Function),
      ),
    );
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("links the fork through the host navigation so an embedded Canvas is not left", async () => {
    // The reader moved on, so the fork is announced rather than opened. The
    // toast link must still route through the owner: a plain browser link would
    // navigate the whole host page out of the embedded Canvas.
    setLocation("/workspace");
    navigationState.currentPath = "/conversations/conv-2";
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockResolvedValue({ id: "fork-1" } as never);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({ eventId: "evt-2" });

    await waitFor(() =>
      expect(displaySuccessToastWithLink).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        "/conversations/fork-1",
        expect.any(Function),
      ),
    );
    expect(navigateMock).not.toHaveBeenCalled();

    // Following the link routes through the navigation owner.
    const onOpen = vi.mocked(displaySuccessToastWithLink).mock.calls[0][3];
    onOpen?.();
    expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1");
  });

  it("surfaces a failed fork as an error toast", async () => {
    vi.mocked(
      AgentServerConversationService.forkConversation,
    ).mockRejectedValue(new Error("fork failed"));

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation("conv-1"), {
      wrapper: Wrapper,
    });

    result.current.mutate({ eventId: "evt-2" });

    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith("fork failed"),
    );
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("refuses to fork without a source conversation", async () => {
    // The Branch action is gated on a conversation, so an empty source only
    // happens on a stale render. Forking would register the mutation under the
    // shared `["fork-conversation", ""]` key and navigate to a fork with no
    // source, so the hook fails fast instead. The toast forwards the error
    // message verbatim, so it must carry translated copy (the I18nKey) rather
    // than an English literal.
    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useForkConversation(""), {
      wrapper: Wrapper,
    });

    result.current.mutate({ eventId: "evt-1" });

    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith(
        i18n.t(I18nKey.CONVERSATION$BRANCH_MISSING_SOURCE),
      ),
    );
    expect(
      AgentServerConversationService.forkConversation,
    ).not.toHaveBeenCalled();
    expect(navigateMock).not.toHaveBeenCalled();
  });
});
