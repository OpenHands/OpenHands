import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

const mocks = vi.hoisted(() => ({
  conversationId: "conversation-1",
  conversation: {
    conversation_url: "http://runtime-1.test",
    session_api_key: "key-1",
    selected_repository: null as string | null,
  },
  runtimeIsReady: true,
  settings: {
    sandbox_grouping_strategy: "NO_GROUPING",
  },
  getGitChanges: vi.fn(),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: mocks.conversationId }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: mocks.conversation }),
}));

vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => mocks.runtimeIsReady,
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: mocks.settings }),
}));

vi.mock("#/api/git-service/v1-git-service.api", () => ({
  default: {
    getGitChanges: mocks.getGitChanges,
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("useUnifiedGetGitChanges", () => {
  beforeEach(() => {
    mocks.conversationId = "conversation-1";
    mocks.conversation = {
      conversation_url: "http://runtime-1.test",
      session_api_key: "key-1",
      selected_repository: null,
    };
    mocks.runtimeIsReady = true;
    mocks.getGitChanges.mockResolvedValue([{ path: "old.ts", status: "M" }]);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("clears ordered changes when the conversation context changes", async () => {
    const { result, rerender } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.data).toEqual([{ path: "old.ts", status: "M" }]);
    });

    mocks.conversationId = "conversation-2";
    mocks.conversation = {
      conversation_url: "http://runtime-2.test",
      session_api_key: "key-2",
      selected_repository: "owner/new-repo",
    };
    mocks.runtimeIsReady = false;

    rerender();

    await waitFor(() => {
      expect(result.current.data).toEqual([]);
    });
  });
});
