import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import { TaskSuggestions } from "#/components/features/home/tasks/task-suggestions";
import { SuggestionsService } from "#/api/suggestions-service/suggestions-service.api";
import { MOCK_TASKS } from "#/mocks/task-suggestions-handlers";

// Mock the translation function
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...(actual as object),
    useTranslation: () => ({
      t: (key: string) => key,
      i18n: {
        changeLanguage: () => new Promise(() => {}),
      },
    }),
  };
});

// Mock the dependencies for useShouldShowUserFeatures
vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({
    data: true,
    isLoading: false,
  }),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: { app_mode: "saas" },
    isLoading: false,
  }),
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({
    providers: [{ id: "github", name: "GitHub" }],
    isLoading: false,
  }),
}));

const renderTaskSuggestions = () => {
  const RouterStub = createRoutesStub([
    {
      Component: () => <TaskSuggestions />,
      path: "/",
    },
    {
      Component: () => <div data-testid="conversation-screen" />,
      path: "/conversations/:conversationId",
    },
    {
      Component: () => <div data-testid="settings-screen" />,
      path: "/settings",
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

describe("TaskSuggestions", () => {
  const getSuggestedTasksSpy = vi.spyOn(
    SuggestionsService,
    "getSuggestedTasks",
  );

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the task suggestions section", () => {
    renderTaskSuggestions();
    screen.getByTestId("task-suggestions");
  });

  it("should render an empty message if there are no tasks", async () => {
    getSuggestedTasksSpy.mockResolvedValue([]);
    renderTaskSuggestions();
    await screen.findByText("TASKS$NO_TASKS_AVAILABLE");
  });

  it("should render the task groups with the correct titles", async () => {
    getSuggestedTasksSpy.mockResolvedValue(MOCK_TASKS);
    renderTaskSuggestions();

    await waitFor(() => {
      // Check for repository names (grouped by repo) - only the first 3 tasks are shown
      screen.getByText("octocat/hello-world");
      screen.getByText("octocat/earth");
    });
  });

  it("should render the task cards with the correct task details", async () => {
    getSuggestedTasksSpy.mockResolvedValue(MOCK_TASKS);
    renderTaskSuggestions();

    await waitFor(() => {
      // Only check for the first 3 tasks that are actually rendered
      // The component limits to 3 tasks due to getLimitedTaskGroups function
      screen.getByText("Fix merge conflicts"); // First task from octocat/hello-world
      screen.getByText("Fix broken CI checks"); // First task from octocat/earth
      screen.getByText("Fix issue"); // Second task from octocat/earth
    });
  });

  it("should render skeletons when loading", async () => {
    getSuggestedTasksSpy.mockResolvedValue(MOCK_TASKS);
    renderTaskSuggestions();

    const skeletons = await screen.findAllByTestId("task-group-skeleton");
    expect(skeletons.length).toBeGreaterThan(0);

    await waitFor(() => {
      // Check for repository names (grouped by repo) - only the first 3 tasks are shown
      screen.getByText("octocat/hello-world");
      screen.getByText("octocat/earth");
    });

    expect(screen.queryByTestId("task-group-skeleton")).not.toBeInTheDocument();
  });
});
