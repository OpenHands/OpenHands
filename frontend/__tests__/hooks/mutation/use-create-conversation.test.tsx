import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { SuggestedTask } from "#/utils/types";

vi.mock("#/hooks/query/use-settings", async () => {
  const actual = await vi.importActual<typeof import("#/hooks/query/use-settings")>(
    "#/hooks/query/use-settings",
  );
  return {
    ...actual,
    useSettings: vi.fn().mockReturnValue({
      data: {
        v1_enabled: true,
      },
      isLoading: false,
    }),
  };
});

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
  }),
}));

describe("useCreateConversation", () => {
  it("passes suggested tasks to the V1 create conversation API", async () => {
    const createConversationSpy = vi
      .spyOn(V1ConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        created_by_user_id: null,
        status: "READY",
        detail: null,
        app_conversation_id: null,
        sandbox_id: null,
        agent_server_url: "http://agent-server.local",
        request: {
          sandbox_id: null,
          initial_message: {
            role: "user",
            content: [{ type: "text", text: "Please address the comments" }],
          },
          processors: [],
          llm_model: null,
          selected_repository: null,
          selected_branch: null,
          git_provider: "github",
          suggested_task: null,
          title: null,
          trigger: null,
          pr_number: [],
          parent_conversation_id: null,
          agent_type: "default",
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

    const { result } = renderHook(() => useCreateConversation(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    });

    const suggestedTask: SuggestedTask = {
      git_provider: "github",
      issue_number: 42,
      repo: "owner/repo",
      title: "Resolve comments",
      task_type: "UNRESOLVED_COMMENTS",
    };

    await result.current.mutateAsync({
      query: "Please address the comments",
      repository: {
        name: "owner/repo",
        gitProvider: "github",
        branch: "main",
      },
      conversationInstructions: "Focus on review comments",
      suggestedTask,
    });

    await waitFor(() => {
      expect(createConversationSpy).toHaveBeenCalledWith(
        "owner/repo",
        "github",
        "Please address the comments",
        "main",
        "Focus on review comments",
        suggestedTask,
        undefined,
        undefined,
        undefined,
      );
    });
  });
});
