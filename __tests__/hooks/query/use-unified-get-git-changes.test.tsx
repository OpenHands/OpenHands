import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import type { GitChange } from "#/api/open-hands.types";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

const getGitChangesSpy = vi.spyOn(AgentServerGitService, "getGitChanges");

const useConversationIdMock = vi.fn<() => { conversationId: string }>();
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => useConversationIdMock(),
  useOptionalConversationId: () => useConversationIdMock(),
}));

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => true,
}));

const conversation = {
  id: "conv-1",
  conversation_url: "https://runtime.example.com/api/conversations/conv-1",
  session_api_key: "session-key",
  workspace: { working_dir: "/workspace/project" },
};

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return {
    client,
    wrapper: function ChangesTestWrapper({
      children,
    }: {
      children: React.ReactNode;
    }) {
      return (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      );
    },
  };
}

function changes(...pairs: Array<[string, GitChange["status"]]>): GitChange[] {
  return pairs.map(([path, status]) => ({ path, status }));
}

beforeEach(() => {
  useConversationIdMock.mockReset();
  useActiveConversationMock.mockReset();
  getGitChangesSpy.mockReset();

  useConversationIdMock.mockReturnValue({ conversationId: "conv-1" });
  useActiveConversationMock.mockReturnValue({ data: conversation });
});

describe("useUnifiedGetGitChanges — ordering is path-identity only", () => {
  it("shows the server's initial order on first load", async () => {
    getGitChangesSpy.mockResolvedValue(changes(["a.ts", "M"], ["b.ts", "A"]));

    const { wrapper } = makeWrapper();
    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(changes(["a.ts", "M"], ["b.ts", "A"]));
  });

  it("reflects a retained path's current status immediately after refetch (M → D)", async () => {
    // First response: `a.ts` is modified.
    getGitChangesSpy.mockResolvedValue(changes(["a.ts", "M"], ["b.ts", "A"]));

    const { wrapper } = makeWrapper();
    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // The same path now shows as deleted; the hook must return the CURRENT
    // object (status D), not the previously-mirrored M object (#16949).
    getGitChangesSpy.mockResolvedValue(changes(["a.ts", "D"], ["b.ts", "A"]));
    await result.current.refetch();

    await waitFor(() =>
      expect(result.current.data).toEqual(
        changes(["a.ts", "D"], ["b.ts", "A"]),
      ),
    );
    // Ordering is preserved even though the response array happens to match.
    expect(result.current.data!.map((c) => c.path)).toEqual(["a.ts", "b.ts"]);
  });

  it("prepends newly-seen paths on top and keeps current objects", async () => {
    getGitChangesSpy.mockResolvedValue(changes(["a.ts", "M"], ["b.ts", "A"]));

    const { wrapper } = makeWrapper();
    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // A brand-new path `c.ts` (untracked) appears; it should land on top of
    // the previously-seen paths.
    getGitChangesSpy.mockResolvedValue(
      changes(["a.ts", "M"], ["b.ts", "A"], ["c.ts", "U"]),
    );
    await result.current.refetch();

    await waitFor(() =>
      expect(result.current.data!.map((c) => c.path)).toEqual([
        "c.ts",
        "a.ts",
        "b.ts",
      ]),
    );
  });

  it("prunes paths that disappear from the response", async () => {
    getGitChangesSpy.mockResolvedValue(changes(["a.ts", "M"], ["b.ts", "A"]));

    const { wrapper } = makeWrapper();
    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // `b.ts` was committed / resolved and no longer appears.
    getGitChangesSpy.mockResolvedValue(changes(["a.ts", "M"]));
    await result.current.refetch();

    await waitFor(() =>
      expect(result.current.data).toEqual(changes(["a.ts", "M"])),
    );
  });

  it("resets path order when the conversation changes", async () => {
    getGitChangesSpy.mockResolvedValue(changes(["a.ts", "M"], ["b.ts", "A"]));

    const { wrapper } = makeWrapper();
    const { result, rerender } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // Switch to another conversation whose workspace has different changes.
    useConversationIdMock.mockReturnValue({ conversationId: "conv-2" });
    useActiveConversationMock.mockReturnValue({
      data: { ...conversation, id: "conv-2" },
    });
    getGitChangesSpy.mockResolvedValue(changes(["x.ts", "U"]));
    rerender();

    await waitFor(() =>
      expect(result.current.data).toEqual(changes(["x.ts", "U"])),
    );
    // Ordering from conv-1 must not leak into conv-2.
    expect(result.current.data!.map((c) => c.path)).toEqual(["x.ts"]);
  });
});
