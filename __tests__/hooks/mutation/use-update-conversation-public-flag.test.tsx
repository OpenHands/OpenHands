import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useUpdateConversationPublicFlag } from "#/hooks/mutation/use-update-conversation-public-flag";

vi.mock("#/api/conversation-service/agent-server-conversation-service.api", () => ({
  default: {
    updateConversationPublicFlag: vi.fn(),
  },
}));

describe("useUpdateConversationPublicFlag", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.clearAllMocks();
  });

  it("optimistically updates all conversation cache entries before the request settles", async () => {
    const conversationId = "conv-1";
    const queryKey = [
      "user",
      "conversation",
      conversationId,
      "cloud-backend",
      "org-1",
    ] as const;

    queryClient.setQueryData(queryKey, {
      conversation_id: conversationId,
      public: false,
    });

    vi.mocked(
      AgentServerConversationService.updateConversationPublicFlag,
    ).mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(
            () =>
              resolve({
                conversation_id: conversationId,
                public: true,
              } as unknown as AppConversation),
            50,
          );
        }),
    );

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(() => useUpdateConversationPublicFlag(), {
      wrapper,
    });

    result.current.mutate({ conversationId, isPublic: true });

    await waitFor(() => {
      expect(queryClient.getQueryData(queryKey)).toMatchObject({
        public: true,
      });
    });
  });
});
