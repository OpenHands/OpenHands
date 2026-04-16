import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import GitService from "#/api/git-service/git-service.api";
import { TaskCard } from "#/components/features/home/tasks/task-card";
import { GitRepository } from "#/types/git";
import { SuggestedTask } from "#/utils/types";

vi.mock("#/hooks/query/use-settings", async () => {
  const actual = await vi.importActual<typeof import("#/hooks/query/use-settings")>(
    "#/hooks/query/use-settings",
  );
  return {
    ...actual,
    getSettingsQueryFn: vi.fn().mockResolvedValue({ v1_enabled: true }),
  };
});

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({ organizationId: null }),
}));

const MOCK_TASK_1: SuggestedTask = {
  issue_number: 123,
  repo: "repo1",
  title: "Task 1",
  task_type: "MERGE_CONFLICTS",
  git_provider: "github",
};

const MOCK_RESPOSITORIES: GitRepository[] = [
  { id: "1", full_name: "repo1", git_provider: "github", is_public: true },
  { id: "2", full_name: "repo2", git_provider: "github", is_public: true },
  { id: "3", full_name: "repo3", git_provider: "gitlab", is_public: true },
  { id: "4", full_name: "repo4", git_provider: "gitlab", is_public: true },
  { id: "5", full_name: "repo5", git_provider: "azure_devops", is_public: true },
];

const renderTaskCard = (task = MOCK_TASK_1) => {
  const RouterStub = createRoutesStub([
    {
      Component: () => <TaskCard task={task} />,
      path: "/",
    },
    {
      Component: () => <div data-testid="conversation-screen" />,
      path: "/conversations/:conversationId",
    },
  ]);

  return render(<RouterStub />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });
};

describe("TaskCard", () => {
  it("format the issue id", async () => {
    renderTaskCard();

    const taskId = screen.getByTestId("task-id");
    expect(taskId).toHaveTextContent(/#123/i);
  });

  it("should call createConversation when clicking the launch button", async () => {
    const createConversationSpy = vi
      .spyOn(V1ConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        created_by_user_id: null,
        status: "READY",
        detail: null,
        app_conversation_id: "conv-123",
        sandbox_id: null,
        agent_server_url: "http://agent-server.local",
        request: {
          sandbox_id: null,
          initial_message: null,
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

    renderTaskCard();

    const launchButton = screen.getByTestId("task-launch-button");
    await userEvent.click(launchButton);

    await waitFor(() => {
      expect(createConversationSpy).toHaveBeenCalled();
    });
  });

  describe("creating suggested task conversation", () => {
    beforeEach(() => {
      const retrieveUserGitRepositoriesSpy = vi.spyOn(
        GitService,
        "retrieveUserGitRepositories",
      );
      retrieveUserGitRepositoriesSpy.mockResolvedValue({
        items: MOCK_RESPOSITORIES,
        next_page_id: null,
      });
    });

    it("should call create conversation with suggest task trigger and selected suggested task", async () => {
      const createConversationSpy = vi
        .spyOn(V1ConversationService, "createConversation")
        .mockResolvedValue({
          id: "task-id",
          created_by_user_id: null,
          status: "READY",
          detail: null,
          app_conversation_id: "conv-123",
          sandbox_id: null,
          agent_server_url: "http://agent-server.local",
          request: {
            sandbox_id: null,
            initial_message: null,
            processors: [],
            llm_model: null,
            selected_repository: MOCK_RESPOSITORIES[0].full_name,
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

      renderTaskCard(MOCK_TASK_1);

      const launchButton = screen.getByTestId("task-launch-button");
      await userEvent.click(launchButton);

      expect(createConversationSpy).toHaveBeenCalledWith(
        MOCK_RESPOSITORIES[0].full_name,
        MOCK_RESPOSITORIES[0].git_provider,
        undefined,
        undefined,
        undefined,
        {
          git_provider: "github",
          issue_number: 123,
          repo: "repo1",
          task_type: "MERGE_CONFLICTS",
          title: "Task 1",
        },
        undefined,
        undefined,
        undefined,
        undefined,
      );
    });
  });

  it("should navigate to the conversation page after creating a conversation", async () => {
    vi.spyOn(V1ConversationService, "createConversation").mockResolvedValue({
      id: "task-id",
      created_by_user_id: null,
      status: "READY",
      detail: null,
      app_conversation_id: "test-conversation-id",
      sandbox_id: null,
      agent_server_url: "http://agent-server.local",
      request: {
        sandbox_id: null,
        initial_message: null,
        processors: [],
        llm_model: null,
        selected_repository: "repo1",
        selected_branch: "main",
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

    renderTaskCard();

    const launchButton = screen.getByTestId("task-launch-button");
    await userEvent.click(launchButton);

    // Wait for navigation to the conversation page
    await screen.findByTestId("conversation-screen");
  });
});
