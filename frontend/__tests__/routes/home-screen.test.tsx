import { render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import { createAxiosNotFoundErrorObject } from "test-utils";
import HomeScreen from "#/routes/home";
import { GitRepository } from "#/types/git";
import SettingsService from "#/api/settings-service/settings-service.api";
import GitService from "#/api/git-service/git-service.api";
import OptionService from "#/api/option-service/option-service.api";
import MainApp from "#/routes/root-layout";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";

const RouterStub = createRoutesStub([
  {
    Component: MainApp,
    path: "/",
    children: [
      {
        Component: HomeScreen,
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
    ],
  },
]);

const selectRepository = async (repoName: string) => {
  const repoConnector = screen.getByTestId("repo-connector");

  // First select the provider
  const providerDropdown = await waitFor(() =>
    screen.getByTestId("git-provider-dropdown"),
  );
  await userEvent.click(providerDropdown);
  await userEvent.click(screen.getByText("GitHub"));

  // Then select the repository
  const repoInput = within(repoConnector).getByTestId("git-repo-dropdown");
  await userEvent.click(repoInput);

  // Wait for the options to be loaded and displayed
  await waitFor(() => {
    const dropdownMenu = screen.getByTestId("git-repo-dropdown-menu");
    expect(within(dropdownMenu).getByText(repoName)).toBeInTheDocument();
  });
  const dropdownMenu = screen.getByTestId("git-repo-dropdown-menu");
  await userEvent.click(within(dropdownMenu).getByText(repoName));

  // Wait for the branch to be auto-selected
  await waitFor(() => {
    const branchInput = screen.getByTestId("git-branch-dropdown-input");
    expect(branchInput).toHaveValue("main");
  });
};

const renderHomeScreen = () =>
  render(<RouterStub />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });

const MOCK_RESPOSITORIES: GitRepository[] = [
  {
    id: "1",
    full_name: "octocat/hello-world",
    git_provider: "github",
    is_public: true,
    main_branch: "main",
  },
  {
    id: "2",
    full_name: "octocat/earth",
    git_provider: "github",
    is_public: true,
    main_branch: "main",
  },
];

describe("HomeScreen", () => {
  beforeEach(() => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {
        github: "fake-token",
        gitlab: "fake-token",
      },
    });
  });

  it("should render", () => {
    renderHomeScreen();
    screen.getByTestId("home-screen");
  });

  it("should render the repository connector and suggested tasks sections", async () => {
    renderHomeScreen();

    await waitFor(() => {
      screen.getByTestId("repo-connector");
      screen.getByTestId("task-suggestions");
    });
  });

  it("should have responsive layout for mobile and desktop screens", async () => {
    renderHomeScreen();

    const homeScreenNewConversationSection = screen.getByTestId(
      "home-screen-new-conversation-section",
    );
    expect(homeScreenNewConversationSection).toHaveClass(
      "flex",
      "flex-col",
      "md:flex-row",
    );

    const homeScreenRecentConversationsSection = screen.getByTestId(
      "home-screen-recent-conversations-section",
    );
    expect(homeScreenRecentConversationsSection).toHaveClass(
      "flex",
      "flex-col",
      "md:flex-row",
    );
  });

  it("should filter the suggested tasks based on the selected repository", async () => {
    const retrieveUserGitRepositoriesSpy = vi.spyOn(
      GitService,
      "retrieveUserGitRepositories",
    );
    retrieveUserGitRepositoriesSpy.mockResolvedValue({
      data: MOCK_RESPOSITORIES,
      nextPage: null,
    });

    // Mock the repository branches API call
    vi.spyOn(GitService, "getRepositoryBranches").mockResolvedValue({
      branches: [
        { name: "main", commit_sha: "123", protected: false },
        { name: "develop", commit_sha: "456", protected: false },
      ],
      has_next_page: false,
      current_page: 1,
      per_page: 30,
      total_count: 2,
    });

    renderHomeScreen();

    const taskSuggestions = await screen.findByTestId("task-suggestions");

    // Initially, all tasks should be visible
    await waitFor(() => {
      within(taskSuggestions).getByText("octocat/hello-world");
      within(taskSuggestions).getByText("octocat/earth");
    });

    // Select a repository using the helper function
    await selectRepository("octocat/hello-world");

    // After selecting a repository, only tasks related to that repository should be visible
    await waitFor(() => {
      within(taskSuggestions).getByText("octocat/hello-world");
      expect(
        within(taskSuggestions).queryByText("octocat/earth"),
      ).not.toBeInTheDocument();
    });
  });

  it("should filter tasks when different repositories are selected", async () => {
    const retrieveUserGitRepositoriesSpy = vi.spyOn(
      GitService,
      "retrieveUserGitRepositories",
    );
    retrieveUserGitRepositoriesSpy.mockResolvedValue({
      data: MOCK_RESPOSITORIES,
      nextPage: null,
    });

    // Mock the repository branches API call
    vi.spyOn(GitService, "getRepositoryBranches").mockResolvedValue({
      branches: [
        { name: "main", commit_sha: "123", protected: false },
        { name: "develop", commit_sha: "456", protected: false },
      ],
      has_next_page: false,
      current_page: 1,
      per_page: 30,
      total_count: 2,
    });

    renderHomeScreen();

    const taskSuggestions = await screen.findByTestId("task-suggestions");

    // Initially, all tasks should be visible
    await waitFor(() => {
      within(taskSuggestions).getByText("octocat/hello-world");
      within(taskSuggestions).getByText("octocat/earth");
    });

    // Select the first repository
    await selectRepository("octocat/hello-world");

    // After selecting first repository, only tasks related to that repository should be visible
    await waitFor(() => {
      within(taskSuggestions).getByText("octocat/hello-world");
      expect(
        within(taskSuggestions).queryByText("octocat/earth"),
      ).not.toBeInTheDocument();
    });

    // Now select the second repository
    await selectRepository("octocat/earth");

    // After selecting second repository, only tasks related to that repository should be visible
    await waitFor(() => {
      within(taskSuggestions).getByText("octocat/earth");
      expect(
        within(taskSuggestions).queryByText("octocat/hello-world"),
      ).not.toBeInTheDocument();
    });
  });

  describe("launch buttons", () => {
    const setupLaunchButtons = async () => {
      let headerLaunchButton = screen.getByTestId(
        "launch-new-conversation-button",
      );
      let repoLaunchButton = await screen.findByTestId("repo-launch-button");
      let tasksLaunchButtons =
        await screen.findAllByTestId("task-launch-button");

      // Mock the repository branches API call
      vi.spyOn(GitService, "getRepositoryBranches").mockResolvedValue({
        branches: [
          { name: "main", commit_sha: "123", protected: false },
          { name: "develop", commit_sha: "456", protected: false },
        ],
        has_next_page: false,
        current_page: 1,
        per_page: 30,
        total_count: 2,
      });

      // Select a repository to enable the repo launch button
      await selectRepository("octocat/hello-world");

      // Wait for all buttons to be enabled
      await waitFor(() => {
        expect(headerLaunchButton).not.toBeDisabled();
        expect(repoLaunchButton).not.toBeDisabled();
        tasksLaunchButtons.forEach((button) => {
          expect(button).not.toBeDisabled();
        });
      });

      headerLaunchButton = screen.getByTestId("launch-new-conversation-button");
      repoLaunchButton = screen.getByTestId("repo-launch-button");
      tasksLaunchButtons = await screen.findAllByTestId("task-launch-button");

      return {
        headerLaunchButton,
        repoLaunchButton,
        tasksLaunchButtons,
      };
    };

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

    it("should disable the other launch buttons when the header launch button is clicked", async () => {
      renderHomeScreen();
      const { headerLaunchButton, repoLaunchButton } =
        await setupLaunchButtons();

      const tasksLaunchButtonsAfter =
        await screen.findAllByTestId("task-launch-button");

      // All other buttons should be disabled when the header button is clicked
      await userEvent.click(headerLaunchButton);

      await waitFor(() => {
        expect(headerLaunchButton).toBeDisabled();
        expect(repoLaunchButton).toBeDisabled();
        tasksLaunchButtonsAfter.forEach((button) => {
          expect(button).toBeDisabled();
        });
      });
    });

    it("should disable the other launch buttons when the repo launch button is clicked", async () => {
      renderHomeScreen();
      const { headerLaunchButton, repoLaunchButton } =
        await setupLaunchButtons();

      const tasksLaunchButtonsAfter =
        await screen.findAllByTestId("task-launch-button");

      // All other buttons should be disabled when the repo button is clicked
      await userEvent.click(repoLaunchButton);

      await waitFor(() => {
        expect(headerLaunchButton).toBeDisabled();
        expect(repoLaunchButton).toBeDisabled();
        tasksLaunchButtonsAfter.forEach((button) => {
          expect(button).toBeDisabled();
        });
      });
    });

    it("should disable the other launch buttons when any task launch button is clicked", async () => {
      renderHomeScreen();
      const { headerLaunchButton, repoLaunchButton, tasksLaunchButtons } =
        await setupLaunchButtons();

      const tasksLaunchButtonsAfter =
        await screen.findAllByTestId("task-launch-button");

      // All other buttons should be disabled when the task button is clicked
      await userEvent.click(tasksLaunchButtons[0]);

      await waitFor(() => {
        expect(headerLaunchButton).toBeDisabled();
        expect(repoLaunchButton).toBeDisabled();
        tasksLaunchButtonsAfter.forEach((button) => {
          expect(button).toBeDisabled();
        });
      });
    });
  });
});

describe("Settings 404", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  const getConfigSpy = vi.spyOn(OptionService, "getConfig");
  const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

  it("should open the settings modal if GET /settings fails with a 404", async () => {
    const error = createAxiosNotFoundErrorObject();
    getSettingsSpy.mockRejectedValue(error);

    renderHomeScreen();

    const settingsModal = await screen.findByTestId("ai-config-modal");
    expect(settingsModal).toBeInTheDocument();
  });

  it("should have the correct advanced settings link that opens in a new window", async () => {
    const error = createAxiosNotFoundErrorObject();
    getSettingsSpy.mockRejectedValue(error);

    renderHomeScreen();

    const settingsScreen = screen.queryByTestId("settings-screen");
    expect(settingsScreen).not.toBeInTheDocument();

    const settingsModal = await screen.findByTestId("ai-config-modal");
    expect(settingsModal).toBeInTheDocument();

    const advancedSettingsLink = await screen.findByTestId(
      "advanced-settings-link",
    );

    // The advanced settings link should be an anchor tag that opens in a new window
    const linkElement = advancedSettingsLink.querySelector("a");
    expect(linkElement).toBeInTheDocument();
    expect(linkElement).toHaveAttribute("href", "/settings");
    expect(linkElement).toHaveAttribute("target", "_blank");
    expect(linkElement).toHaveAttribute("rel", "noreferrer noopener");
  });

  it("should not open the settings modal if GET /settings fails but is SaaS mode", async () => {
    // @ts-expect-error - we only need APP_MODE for this test
    getConfigSpy.mockResolvedValue({
      APP_MODE: "saas",
      FEATURE_FLAGS: {
        ENABLE_BILLING: false,
        HIDE_LLM_SETTINGS: false,
        ENABLE_JIRA: false,
        ENABLE_JIRA_DC: false,
        ENABLE_LINEAR: false,
      },
    });
    const error = createAxiosNotFoundErrorObject();
    getSettingsSpy.mockRejectedValue(error);

    renderHomeScreen();

    expect(screen.queryByTestId("ai-config-modal")).not.toBeInTheDocument();
  });
});

describe("Setup Payment modal", () => {
  const getConfigSpy = vi.spyOn(OptionService, "getConfig");
  const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

  it("should only render if SaaS mode and is new user", async () => {
    // @ts-expect-error - we only need the APP_MODE for this test
    getConfigSpy.mockResolvedValue({
      APP_MODE: "saas",
      FEATURE_FLAGS: {
        ENABLE_BILLING: true,
        HIDE_LLM_SETTINGS: false,
        ENABLE_JIRA: false,
        ENABLE_JIRA_DC: false,
        ENABLE_LINEAR: false,
      },
    });
    const error = createAxiosNotFoundErrorObject();
    getSettingsSpy.mockRejectedValue(error);

    renderHomeScreen();

    const setupPaymentModal = await screen.findByTestId(
      "proceed-to-stripe-button",
    );
    expect(setupPaymentModal).toBeInTheDocument();
  });
});
