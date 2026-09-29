import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { NavigationProvider } from "#/context/navigation-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

const conversation: AppConversation = {
  id: "conv-1",
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Test",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  execution_status: null,
  conversation_url: "https://old.example.com",
  session_api_key: "old-key",
  sandbox_id: null,
  workspace: { working_dir: "/old" },
  sub_conversation_ids: [],
};

afterEach(() => {
  vi.restoreAllMocks();
  ConversationService.setCurrentConversation(null);
});

it("updates the service when a refetch changes runtime details but not execution status", async () => {
  // Arrange: the same conversation is still idle after its runtime changes.
  const updatedConversation: AppConversation = {
    ...conversation,
    conversation_url: "https://new.example.com",
    session_api_key: "new-key",
    workspace: { working_dir: "/new" },
  };
  vi.spyOn(AgentServerConversationService, "batchGetAppConversations")
    .mockResolvedValueOnce([conversation])
    .mockResolvedValueOnce([updatedConversation]);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <NavigationProvider
        value={{
          currentPath: "/conversations/conv-1",
          conversationId: "conv-1",
          isNavigating: false,
          navigate: vi.fn(),
        }}
      >
        {children}
      </NavigationProvider>
    </QueryClientProvider>
  );
  const { result } = renderHook(() => useActiveConversation(), { wrapper });
  await waitFor(() =>
    expect(ConversationService.getCurrentConversation()).toEqual(conversation),
  );

  // Act: the query receives fresh runtime details without a status transition.
  await result.current.refetch();

  // Assert: downstream uploads and runtime clients use the latest details.
  await waitFor(() =>
    expect(ConversationService.getCurrentConversation()).toEqual(
      updatedConversation,
    ),
  );
});
