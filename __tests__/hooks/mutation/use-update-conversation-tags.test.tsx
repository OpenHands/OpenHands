import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useUpdateConversationTags } from "#/hooks/mutation/use-update-conversation-tags";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

// Mock the AgentServerConversationService
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      updateConversationTags: vi.fn(),
    },
  }),
);

const createWrapper = (queryClient?: QueryClient) => {
  const client =
    queryClient ??
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
        mutations: {
          retry: false,
        },
      },
    });

  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );

  return { Wrapper, client };
};

describe("useUpdateConversationTags", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls updateConversationTags with the conversation id and merged tags", async () => {
    vi.mocked(
      AgentServerConversationService.updateConversationTags,
    ).mockResolvedValue({
      id: "test-conversation-id",
      tags: { origin: "slack" },
    } as unknown as AppConversation);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useUpdateConversationTags(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId: "test-conversation-id",
      tags: { origin: "slack" },
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(
      AgentServerConversationService.updateConversationTags,
    ).toHaveBeenCalledWith("test-conversation-id", { origin: "slack" });
  });

  it("optimistically updates the prefix-keyed conversation cache and rolls back on error", async () => {
    const conversationId = "test-conversation-id";
    // Mirror the key shape from `useUserConversation`:
    //   ["user", "conversation", cid, backendId, orgId]
    const cacheKey = [
      "user",
      "conversation",
      conversationId,
      "default-local",
      null,
    ] as const;
    const baseConversation = {
      id: conversationId,
      tags: { origin: "slack", acpserver: "claude-code" },
    } as unknown as AppConversation;

    vi.mocked(
      AgentServerConversationService.updateConversationTags,
    ).mockRejectedValue(new Error("nope"));

    const { Wrapper, client } = createWrapper();
    client.setQueryData(cacheKey, baseConversation);

    const { result } = renderHook(() => useUpdateConversationTags(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId,
      tags: { origin: "irc", acpserver: "claude-code" },
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    const cached = client.getQueryData<AppConversation>(cacheKey);
    expect(cached?.tags).toEqual({
      origin: "slack",
      acpserver: "claude-code",
    });
  });

  it("invalidates user-conversation and user-conversations on settle", async () => {
    const conversationId = "test-conversation-id";

    vi.mocked(
      AgentServerConversationService.updateConversationTags,
    ).mockResolvedValue({
      id: conversationId,
      tags: { origin: "slack" },
    } as unknown as AppConversation);

    const { Wrapper, client } = createWrapper();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useUpdateConversationTags(), {
      wrapper: Wrapper,
    });

    await result.current.mutateAsync({
      conversationId,
      tags: { origin: "slack" },
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversation", conversationId],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversations"],
      });
    });
  });
});
