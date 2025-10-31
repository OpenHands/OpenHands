import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import UserService from "#/api/user-service/user-service.api";
import GitService from "#/api/git-service/git-service.api";
import { TaskCard } from "#/components/features/home/tasks/task-card";
import { GitRepository } from "#/types/git";
import { SuggestedTask } from "#/utils/types";

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
    const createConversationSpy = vi.spyOn(
      ConversationService,
      "createConversation",
    );

    renderTaskCard();

    const launchButton = screen.getByTestId("task-launch-button");
    await userEvent.click(launchButton);

    expect(createConversationSpy).toHaveBeenCalled();
  });

  describe("creating suggested task conversation", () => {
    beforeEach(() => {
      const retrieveUserGitRepositoriesSpy = vi.spyOn(
        GitService,
        "retrieveUserGitRepositories",
      );
      retrieveUserGitRepositoriesSpy.mockResolvedValue({
        data: MOCK_RESPOSITORIES,
        nextPage: null,
      });
    });

    it("should call create conversation with suggest task trigger and selected suggested task", async () => {
      const createConversationSpy = vi.spyOn(
        ConversationService,
        "createConversation",
      );

      renderTaskCard(MOCK_TASK_1);

      const launchButton = screen.getByTestId("task-launch-button");
      await userEvent.click(launchButton);

      expect(createConversationSpy).toHaveBeenCalledWith(
        MOCK_RESPOSITORIES[0].full_name,
        MOCK_RESPOSITORIES[0].git_provider,
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
      );
    });
  });

  it("should navigate to the conversation page after creating a conversation", async () => {
    const createConversationSpy = vi.spyOn(
      ConversationService,
      "createConversation",
    );
    createConversationSpy.mockResolvedValue({
      conversation_id: "test-conversation-id",
      title: "Test Conversation",
      selected_repository: "repo1",
      selected_branch: "main",
      git_provider: "github",
      last_updated_at: "2023-01-01T00:00:00Z",
      created_at: "2023-01-01T00:00:00Z",
      status: "RUNNING",
      runtime_status: "STATUS$READY",
      url: null,
      session_api_key: null,
    });

    renderTaskCard();

    const launchButton = screen.getByTestId("task-launch-button");
    await userEvent.click(launchButton);

    // Wait for navigation to the conversation page
    await screen.findByTestId("conversation-screen");
  });
});
