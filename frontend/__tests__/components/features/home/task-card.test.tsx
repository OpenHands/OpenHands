import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import { TaskCard } from "#/components/features/home/tasks/task-card";
import { SuggestedTask } from "#/utils/types";

const mockCreateConversation = vi.fn();

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: mockCreateConversation,
  }),
}));

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
  beforeEach(() => {
    mockCreateConversation.mockReset();
  });

  it("format the issue id", async () => {
    renderTaskCard();

    const taskId = screen.getByTestId("task-id");
    expect(taskId).toHaveTextContent(/#123/i);
  });

  it("should call createConversation when clicking the launch button", async () => {
    renderTaskCard();

    const launchButton = screen.getByTestId("task-launch-button");
    await userEvent.click(launchButton);

    await waitFor(() => {
      expect(mockCreateConversation).toHaveBeenCalled();
    });
  });

  describe("creating suggested task conversation", () => {
    it("should call create conversation with suggest task trigger and selected suggested task", async () => {
      renderTaskCard(MOCK_TASK_1);

      const launchButton = screen.getByTestId("task-launch-button");
      await userEvent.click(launchButton);

      expect(mockCreateConversation).toHaveBeenCalledWith(
        {
          repository: {
            name: "repo1",
            gitProvider: "github",
          },
          suggestedTask: MOCK_TASK_1,
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
        }),
      );
    });
  });

  it("should navigate to the conversation page after creating a conversation", async () => {
    mockCreateConversation.mockImplementation((_variables, options) => {
      options?.onSuccess?.({
        conversation_id: "test-conversation-id",
      });
    });

    renderTaskCard();

    const launchButton = screen.getByTestId("task-launch-button");
    await userEvent.click(launchButton);

    // Wait for navigation to the conversation page
    await screen.findByTestId("conversation-screen");
  });
});
