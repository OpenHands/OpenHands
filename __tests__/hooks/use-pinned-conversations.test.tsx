import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PINNED_TAG_KEY } from "#/api/agent-server-adapter";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { usePinnedConversations } from "#/hooks/use-pinned-conversations";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";

const useActiveBackendMock = vi.fn();
const mergeConversationTagsMock = vi.fn();

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      mergeConversationTags: (
        ...args: Parameters<typeof mergeConversationTagsMock>
      ) => mergeConversationTagsMock(...args),
    },
  }),
);
vi.mock("@tanstack/react-query", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@tanstack/react-query")>()),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
  useMutation: ({
    mutationFn,
  }: {
    mutationFn: (variables: unknown) => Promise<unknown>;
  }) => ({
    mutate: (variables: unknown) => {
      void mutationFn(variables);
    },
  }),
}));

const BACKEND_ID = "default-local";

function conversation(
  id: string,
  tags: Record<string, string> | null = null,
): AppConversation {
  return { id, tags } as AppConversation;
}

describe("usePinnedConversations", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    usePinnedConversationsStore.setState({ pinsByBackendId: {} });
    mergeConversationTagsMock.mockResolvedValue(undefined);
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "local", id: BACKEND_ID },
    });
  });

  it("orders pinned conversations most recently pinned first", () => {
    const conversations = [
      conversation("a", { [PINNED_TAG_KEY]: "2026-01-01T00:00:00.000Z" }),
      conversation("b"),
      conversation("c", { [PINNED_TAG_KEY]: "2026-03-01T00:00:00.000Z" }),
      conversation("d", { [PINNED_TAG_KEY]: "2026-02-01T00:00:00.000Z" }),
    ];

    const { result } = renderHook(() => usePinnedConversations(conversations));

    expect(result.current.pinnedIds).toEqual(["c", "d", "a"]);
  });

  it("writes a pinned tag when pinning and clears it when unpinning", () => {
    const conversations = [
      conversation("a"),
      conversation("b", { [PINNED_TAG_KEY]: "2026-01-01T00:00:00.000Z" }),
    ];

    const { result } = renderHook(() => usePinnedConversations(conversations));

    result.current.togglePin("a");
    expect(mergeConversationTagsMock).toHaveBeenCalledWith("a", {
      [PINNED_TAG_KEY]: expect.any(String),
    });

    result.current.togglePin("b");
    expect(mergeConversationTagsMock).toHaveBeenCalledWith("b", {
      [PINNED_TAG_KEY]: null,
    });
  });

  it("sends only the pin key, leaving the tag merge to the service", () => {
    const conversations = [conversation("a", { acpserver: "gemini-cli" })];

    const { result } = renderHook(() => usePinnedConversations(conversations));
    result.current.togglePin("a");

    expect(mergeConversationTagsMock).toHaveBeenCalledTimes(1);
    expect(Object.keys(mergeConversationTagsMock.mock.calls[0][1])).toEqual([
      PINNED_TAG_KEY,
    ]);
  });

  it("replays pins made before the migration onto the backend, then drops them locally", async () => {
    usePinnedConversationsStore.setState({
      pinsByBackendId: { [BACKEND_ID]: ["b", "a"] },
    });

    renderHook(() =>
      usePinnedConversations([conversation("a"), conversation("b")]),
    );

    await waitFor(() =>
      expect(mergeConversationTagsMock).toHaveBeenCalledTimes(2),
    );
    // "b" was pinned most recently locally, so it must end up with the newest
    // timestamp and stay at the top of the pinned section.
    const stamped = Object.fromEntries(
      mergeConversationTagsMock.mock.calls.map(([id, patch]) => [
        id,
        patch[PINNED_TAG_KEY],
      ]),
    );
    expect(stamped.b > stamped.a).toBe(true);

    await waitFor(() =>
      expect(
        usePinnedConversationsStore.getState().pinsByBackendId[BACKEND_ID],
      ).toEqual([]),
    );
  });

  it("does not replay local pins for conversations the backend no longer has", async () => {
    usePinnedConversationsStore.setState({
      pinsByBackendId: { [BACKEND_ID]: ["gone", "a"] },
    });

    renderHook(() => usePinnedConversations([conversation("a")]));

    await waitFor(() =>
      expect(mergeConversationTagsMock).toHaveBeenCalledTimes(1),
    );
    expect(mergeConversationTagsMock.mock.calls[0][0]).toBe("a");
  });

  it("waits for the conversation list before migrating, so no pin is discarded", () => {
    usePinnedConversationsStore.setState({
      pinsByBackendId: { [BACKEND_ID]: ["a"] },
    });

    renderHook(() => usePinnedConversations([]));

    expect(mergeConversationTagsMock).not.toHaveBeenCalled();
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[BACKEND_ID],
    ).toEqual(["a"]);
  });

  it("keeps cloud backends on the browser-local store", () => {
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "cloud", id: "cloud-prod" },
    });

    const { result } = renderHook(() =>
      usePinnedConversations([conversation("a")]),
    );

    result.current.togglePin("a");

    expect(mergeConversationTagsMock).not.toHaveBeenCalled();
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId["cloud-prod"],
    ).toEqual(["a"]);
  });
});
