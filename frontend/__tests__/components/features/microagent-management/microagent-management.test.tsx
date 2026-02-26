import { screen, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClientConfig } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import React from "react";
import { renderWithProviders } from "test-utils";
import MicroagentManagement from "#/routes/microagent-management";
import { MicroagentManagementMain } from "#/components/features/microagent-management/microagent-management-main";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import GitService from "#/api/git-service/git-service.api";
import { GitRepository } from "#/types/git";
import { RepositoryMicroagent } from "#/types/microagent-management";
import { Conversation } from "#/api/open-hands.types";
import { useMicroagentManagementStore } from "#/stores/microagent-management-store";

// Mock hooks
const mockUseUserProviders = vi.fn();
const mockUseGitRepositories = vi.fn();
const mockUseConfig = vi.fn();
const mockUseRepositoryMicroagents = vi.fn();
const mockUseMicroagentManagementConversations = vi.fn();
const mockUseSearchRepositories = vi.fn();
const mockUseCreateConversationAndSubscribeMultiple = vi.fn();

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => mockUseUserProviders(),
}));

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: () => mockUseGitRepositories(),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

vi.mock("#/hooks/query/use-repository-microagents", () => ({
  useRepositoryMicroagents: () => mockUseRepositoryMicroagents(),
}));

vi.mock("#/hooks/query/use-microagent-management-conversations", () => ({
  useMicroagentManagementConversations: () =>
    mockUseMicroagentManagementConversations(),
}));

vi.mock("#/hooks/query/use-search-repositories", () => ({
  useSearchRepositories: () => mockUseSearchRepositories(),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackEvent: vi.fn(),
  }),
}));

vi.mock("#/hooks/use-create-conversation-and-subscribe-multiple", () => ({
  useCreateConversationAndSubscribeMultiple: () =>
    mockUseCreateConversationAndSubscribeMultiple(),
}));

describe("MicroagentManagement", () => {
  const RouterStub = createRoutesStub([
    {
      Component: MicroagentManagement,
      path: "/",
    },
  ]);

  const renderMicroagentManagement = (config?: QueryClientConfig) =>
    renderWithProviders(<RouterStub />);

  // Common test data
  const testRepository = {
    id: "1",
    full_name: "user/test-repo",
    git_provider: "github" as const,
    is_public: true,
    owner_type: "user" as const,
    pushed_at: "2021-10-01T12:00:00Z",
  };

  // Helper function to render with custom Zustand store state
  const renderWithCustomStore = (storeOverrides: Partial<any>) => {
    useMicroagentManagementStore.setState(storeOverrides);
    return renderWithProviders(<RouterStub />);
  };

  // Helper function to render with update modal visible
  const renderWithUpdateModal = (additionalState: Partial<any> = {}) => {
    return renderWithCustomStore({
      updateMicroagentModalVisible: true,
      selectedRepository: testRepository,
      ...additionalState,
    });
  };

  // Helper function to render with selected microagent
  const renderWithSelectedMicroagent = (
    microagent: any,
    additionalState: Partial<any> = {},
  ) => {
    return renderWithCustomStore({
      selectedRepository: testRepository,
      selectedMicroagentItem: {
        microagent,
        conversation: null,
      },
      ...additionalState,
    });
  };

  beforeAll(() => {
    vi.mock("react-router", async (importOriginal) => ({
      ...(await importOriginal<typeof import("react-router")>()),
      Link: ({ children }: React.PropsWithChildren) => children,
      useNavigate: vi.fn(() => vi.fn()),
      useLocation: vi.fn(() => ({ pathname: "/microagent-management" })),
      useParams: vi.fn(() => ({ conversationId: "2" })),
    }));
  });

  const mockRepositories: GitRepository[] = [
    {
      id: "1",
      full_name: "user/repo1",
      git_provider: "github",
      is_public: true,
      owner_type: "user",
      pushed_at: "2021-10-01T12:00:00Z",
    },
    {
      id: "2",
      full_name: "user/repo2/.openhands",
      git_provider: "github",
      is_public: true,
      owner_type: "user",
      pushed_at: "2021-10-02T12:00:00Z",
    },
    {
      id: "3",
      full_name: "org/repo3/.openhands",
      git_provider: "github",
      is_public: true,
      owner_type: "organization",
      pushed_at: "2021-10-03T12:00:00Z",
    },
    {
      id: "4",
      full_name: "user/repo4",
      git_provider: "github",
      is_public: true,
      owner_type: "user",
      pushed_at: "2021-10-04T12:00:00Z",
    },
    {
      id: "5",
      full_name: "user/TestRepository",
      git_provider: "github",
      is_public: true,
      owner_type: "user",
      pushed_at: "2021-10-05T12:00:00Z",
    },
    {
      id: "6",
      full_name: "org/AnotherRepo",
      git_provider: "github",
      is_public: true,
      owner_type: "organization",
      pushed_at: "2021-10-06T12:00:00Z",
    },
    {
      id: "7",
      full_name: "user/gitlab-repo/openhands-config",
      git_provider: "gitlab",
      is_public: true,
      owner_type: "user",
      pushed_at: "2021-10-07T12:00:00Z",
    },
    {
      id: "8",
      full_name: "org/gitlab-org-repo/openhands-config",
      git_provider: "gitlab",
      is_public: true,
      owner_type: "organization",
      pushed_at: "2021-10-08T12:00:00Z",
    },
  ];

  // Helper function to filter repositories with OpenHands suffixes
  const getRepositoriesWithOpenHandsSuffix = (
    repositories: GitRepository[],
  ) => {
    return repositories.filter(
      (repo) =>
        repo.full_name.endsWith("/.openhands") ||
        repo.full_name.endsWith("/openhands-config"),
    );
  };

  // Helper functions for mocking search repositories
  const mockSearchRepositoriesWithData = (data: GitRepository[]) => {
    mockUseSearchRepositories.mockReturnValue({
      data,
      isLoading: false,
      isError: false,
    });
  };

  const mockSearchRepositoriesEmpty = () => {
    mockUseSearchRepositories.mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
    });
  };

  const mockMicroagents: RepositoryMicroagent[] = [
    {
      name: "test-microagent-1",
      created_at: "2021-10-01T12:00:00Z",
      git_provider: "github",
      path: ".openhands/microagents/test-microagent-1",
    },
    {
      name: "test-microagent-2",
      created_at: "2021-10-02T12:00:00Z",
      git_provider: "github",
      path: ".openhands/microagents/test-microagent-2",
    },
  ];

  const mockConversations: Conversation[] = [
    {
      conversation_id: "conv-1",
      title: "Test Conversation 1",
      selected_repository: "user/repo2/.openhands",
      selected_branch: "main",
      git_provider: "github",
      last_updated_at: "2021-10-01T12:00:00Z",
      created_at: "2021-10-01T12:00:00Z",
      status: "RUNNING",
      runtime_status: null,
      trigger: "microagent_management",
      url: null,
      session_api_key: null,
    },
    {
      conversation_id: "conv-2",
      title: "Test Conversation 2",
      selected_repository: "user/repo2/.openhands",
      selected_branch: "main",
      git_provider: "github",
      last_updated_at: "2021-10-02T12:00:00Z",
      created_at: "2021-10-02T12:00:00Z",
      status: "STOPPED",
      runtime_status: null,
      trigger: "microagent_management",
      url: null,
      session_api_key: null,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();

    // Reset Zustand store to default state
    useMicroagentManagementStore.setState({
      // Modal visibility states
      addMicroagentModalVisible: false,
      updateMicroagentModalVisible: false,
      learnThisRepoModalVisible: false,

      // Repository states
      selectedRepository: null,
      personalRepositories: [],
      organizationRepositories: [],
      repositories: [],

      // Microagent states
      selectedMicroagentItem: null,
    });

    // Setup default hook mocks
    mockUseUserProviders.mockReturnValue({
      providers: ["github"],
    });

    mockUseGitRepositories.mockReturnValue({
      data: {
        pages: [
          {
            data: mockRepositories,
            nextPage: null,
          },
        ],
      },
      isLoading: false,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      onLoadMore: vi.fn(),
    });

    mockUseConfig.mockReturnValue({
      data: {
        app_mode: "oss",
      },
    });

    mockUseRepositoryMicroagents.mockReturnValue({
      data: mockMicroagents,
      isLoading: false,
      isError: false,
    });

    mockUseMicroagentManagementConversations.mockReturnValue({
      data: mockConversations,
      isLoading: false,
      isError: false,
    });

    mockUseCreateConversationAndSubscribeMultiple.mockReturnValue({
      createConversationAndSubscribe: vi.fn(({ onSuccessCallback }) => {
        // Immediately call the success callback to close the modal
        if (onSuccessCallback) {
          onSuccessCallback();
        }
      }),
      isPending: false,
    });

    // Mock the search repositories hook to return repositories with OpenHands suffixes
    const mockSearchResults =
      getRepositoriesWithOpenHandsSuffix(mockRepositories);

    mockSearchRepositoriesWithData(mockSearchResults);

    // Setup default mock for retrieveUserGitRepositories
    vi.spyOn(GitService, "retrieveUserGitRepositories").mockResolvedValue({
      data: [...mockRepositories],
      nextPage: null,
    });
    // Setup default mock for getRepositoryMicroagents
    vi.spyOn(GitService, "getRepositoryMicroagents").mockResolvedValue([
      ...mockMicroagents,
    ]);
    // Setup default mock for searchConversations
    vi.spyOn(ConversationService, "searchConversations").mockResolvedValue([
      ...mockConversations,
    ]);
    // Setup default mock for getRepositoryMicroagentContent
    vi.spyOn(GitService, "getRepositoryMicroagentContent").mockResolvedValue({
      content: "Original microagent content for testing updates",
      path: ".openhands/microagents/update-test-microagent",
      git_provider: "github",
      triggers: ["test", "update"],
    });
  });

  it("should render the microagent management page", async () => {
    renderMicroagentManagement();

    // Check that the main title is rendered
    await screen.findByText("MICROAGENT_MANAGEMENT$DESCRIPTION");
  });

  it("should display loading state when fetching repositories", async () => {
    // Mock loading state
    mockUseGitRepositories.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      onLoadMore: vi.fn(),
    });

    renderMicroagentManagement();

    // Check that loading spinner is displayed
    const loadingSpinner = await screen.findByText("HOME$LOADING_REPOSITORIES");
    expect(loadingSpinner).toBeInTheDocument();
  });

  it("should handle error when fetching repositories", async () => {
    // Mock error state
    mockUseGitRepositories.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      hasNextPage: false,
      isFetchingNextPage: false,
      onLoadMore: vi.fn(),
    });

    renderMicroagentManagement();

    // Wait for the error to be handled
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });
  });

  it("should categorize repositories correctly", async () => {
    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Check that tabs are rendered
    const personalTab = screen.getByText("COMMON$PERSONAL");
    const repositoriesTab = screen.getByText("COMMON$REPOSITORIES");
    const organizationsTab = screen.getByText("COMMON$ORGANIZATIONS");

    expect(personalTab).toBeInTheDocument();
    expect(repositoriesTab).toBeInTheDocument();
    expect(organizationsTab).toBeInTheDocument();
  });

  it("should display repositories in accordion", async () => {
    renderMicroagentManagement();

    // Wait for repositories to be loaded and rendered
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Check that repository names are displayed
    const repo1 = screen.getByTestId("repository-name-tooltip");
    expect(repo1).toBeInTheDocument();
    expect(repo1).toHaveTextContent("user/repo2/.openhands");
  });

  it("should expand repository accordion and show microagents", async () => {
    const user = userEvent.setup();
    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Find and click on the first repository accordion
    const repoAccordion = screen.getByTestId("repository-name-tooltip");
    await user.click(repoAccordion);

    // Wait for microagents to be fetched
    await waitFor(() => {
      expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
    });

    // Check that microagents are displayed
    const microagent1 = screen.getByText("test-microagent-1");
    const microagent2 = screen.getByText("test-microagent-2");

    expect(microagent1).toBeInTheDocument();
    expect(microagent2).toBeInTheDocument();
  });

  it("should display loading state when fetching microagents", async () => {
    const user = userEvent.setup();
    mockUseRepositoryMicroagents.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
    });

    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Find and click on the first repository accordion
    const repoAccordion = screen.getByTestId("repository-name-tooltip");
    await user.click(repoAccordion);

    // Check that loading spinner is displayed
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("should handle error when fetching microagents", async () => {
    const user = userEvent.setup();
    mockUseRepositoryMicroagents.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    });

    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Find and click on the first repository accordion
    const repoAccordion = screen.getByTestId("repository-name-tooltip");
    await user.click(repoAccordion);

    // Wait for the error to be handled
    await waitFor(() => {
      expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
    });
  });

  it("should display empty state when no microagents are found", async () => {
    const user = userEvent.setup();
    mockUseRepositoryMicroagents.mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
    });

    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Find and click on the first repository accordion
    const repoAccordion = screen.getByTestId("repository-name-tooltip");
    await user.click(repoAccordion);

    // Wait for microagents to be fetched
    await waitFor(() => {
      expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
    });

    // Check that no microagents are displayed
    expect(screen.queryByText("test-microagent-1")).not.toBeInTheDocument();
    expect(screen.queryByText("test-microagent-2")).not.toBeInTheDocument();
  });

  it("should display microagent cards with correct information", async () => {
    const user = userEvent.setup();
    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Find and click on the first repository accordion
    const repoAccordion = screen.getByTestId("repository-name-tooltip");
    await user.click(repoAccordion);

    // Wait for microagents to be fetched
    await waitFor(() => {
      expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
    });

    // Check that microagent cards display correct information
    const microagent1 = screen.getByText("test-microagent-1");
    const microagent2 = screen.getByText("test-microagent-2");

    expect(microagent1).toBeInTheDocument();
    expect(microagent2).toBeInTheDocument();

    // Check that microagent file paths are displayed
    const filePath1 = screen.getByText(
      ".openhands/microagents/test-microagent-1",
    );
    const filePath2 = screen.getByText(
      ".openhands/microagents/test-microagent-2",
    );

    expect(filePath1).toBeInTheDocument();
    expect(filePath2).toBeInTheDocument();
  });

  it("should render add microagent button", async () => {
    renderMicroagentManagement();

    // Wait for repositories to be loaded and processed
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Wait for repositories to be displayed in the accordion
    await waitFor(() => {
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
    });

    // Check that add microagent buttons are present
    const addButtons = screen.getAllByTestId("add-microagent-button");
    expect(addButtons.length).toBeGreaterThan(0);
  });

  it("should open modal when add button is clicked", async () => {
    const user = userEvent.setup();
    renderMicroagentManagement();

    // Wait for repositories to be loaded and processed
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Wait for repositories to be displayed in the accordion
    await waitFor(() => {
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
    });

    // Find and click the first add microagent button
    const addButtons = screen.getAllByTestId("add-microagent-button");
    await user.click(addButtons[0]);

    // Check that the modal is opened
    await waitFor(() => {
      expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
    });
  });

  it("should close add microagent modal when cancel is clicked", async () => {
    const user = userEvent.setup();
    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Find and click the first add microagent button
    const addButtons = screen.getAllByTestId("add-microagent-button");
    await user.click(addButtons[0]);

    // Check that the modal is opened
    const closeButton = screen.getByRole("button", { name: "" });
    await user.click(closeButton);

    // Check that modal is closed
    await waitFor(() => {
      expect(
        screen.queryByTestId("add-microagent-modal"),
      ).not.toBeInTheDocument();
    });
  });

  it("should display empty state when no repositories are found", async () => {
    // Mock empty repositories
    mockUseGitRepositories.mockReturnValue({
      data: {
        pages: [
          {
            data: [],
            nextPage: null,
          },
        ],
      },
      isLoading: false,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      onLoadMore: vi.fn(),
    });

    // Mock empty search results
    mockSearchRepositoriesEmpty();

    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Check that empty state messages are displayed
    const personalEmptyState = screen.getByText(
      "MICROAGENT_MANAGEMENT$YOU_DO_NOT_HAVE_USER_LEVEL_MICROAGENTS",
    );

    expect(personalEmptyState).toBeInTheDocument();
  });

  it("should handle multiple repository expansions", async () => {
    const user = userEvent.setup();
    renderMicroagentManagement();

    // Wait for repositories to be loaded
    await waitFor(() => {
      expect(mockUseGitRepositories).toHaveBeenCalled();
    });

    // Find and click on the first repository accordion
    const repoAccordion1 = screen.getByTestId("repository-name-tooltip");
    await user.click(repoAccordion1);

    // Wait for microagents to be fetched for first repo
    await waitFor(() => {
      expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
    });

    // Check that the hook was called
    expect(mockUseRepositoryMicroagents).toHaveBeenCalledTimes(1);
  });

  it("should display ready to add microagent message in main area", async () => {
    renderMicroagentManagement();

    // Check that the main area shows the ready message
    const readyMessage = screen.getByText(
      "MICROAGENT_MANAGEMENT$READY_TO_ADD_MICROAGENT",
    );
    const descriptionMessage = screen.getByText(
      "MICROAGENT_MANAGEMENT$OPENHANDS_CAN_LEARN_ABOUT_REPOSITORIES",
    );

    expect(readyMessage).toBeInTheDocument();
    expect(descriptionMessage).toBeInTheDocument();
  });

  // Search functionality tests
  describe("Search functionality", () => {
    it("should render search input field", async () => {
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Check that search input is rendered
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      expect(searchInput).toBeInTheDocument();
      expect(searchInput).toHaveAttribute(
        "placeholder",
        "COMMON$SEARCH_REPOSITORIES...",
      );
    });

    it("should filter repositories when typing in search input", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Initially only repositories with .openhands should be visible
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
      expect(screen.queryByText("user/repo4")).not.toBeInTheDocument();

      // Type in search input to filter further
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, "repo2");

      // Only repo2 should be visible
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
      expect(
        screen.queryByText("org/repo3/.openhands"),
      ).not.toBeInTheDocument();
      expect(screen.queryByText("user/repo4")).not.toBeInTheDocument();
      expect(screen.queryByText("user/TestRepository")).not.toBeInTheDocument();
      expect(screen.queryByText("org/AnotherRepo")).not.toBeInTheDocument();
    });

    it("should perform case-insensitive search", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Type in search input with uppercase
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, "REPO2");

      // repo2 should be visible (case-insensitive match)
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
      expect(
        screen.queryByText("org/repo3/.openhands"),
      ).not.toBeInTheDocument();
    });

    it("should filter repositories by partial matches", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Type in search input with partial match
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, "repo");

      // All repositories with "repo" in the name should be visible
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
      expect(
        screen.queryByText("org/repo3/.openhands"),
      ).not.toBeInTheDocument();
      expect(screen.queryByText("user/repo4")).not.toBeInTheDocument();
      expect(screen.queryByText("user/TestRepository")).not.toBeInTheDocument();
      expect(screen.queryByText("org/AnotherRepo")).not.toBeInTheDocument();
    });

    it("should show all repositories when search input is cleared", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Type in search input
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, "repo2");

      // Only repo2 should be visible
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();

      // Clear the search input
      await user.clear(searchInput);

      // All repositories should be visible again (only those with .openhands)
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
      expect(
        screen.queryByText("org/repo3/.openhands"),
      ).not.toBeInTheDocument();
      expect(screen.queryByText("user/repo4")).not.toBeInTheDocument();
      expect(screen.queryByText("user/TestRepository")).not.toBeInTheDocument();
      expect(screen.queryByText("org/AnotherRepo")).not.toBeInTheDocument();
    });

    it("should handle empty search results", async () => {
      const user = userEvent.setup();

      // Mock empty search results for this test
      mockSearchRepositoriesEmpty();

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Type in search input with non-existent repository name
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, "nonexistent");

      // Wait for debounced search to complete (300ms debounce + buffer)
      await new Promise((resolve) => setTimeout(resolve, 400));

      // Wait for the search to complete and check that no repositories are visible
      await waitFor(() => {
        expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
        expect(
          screen.queryByText("user/repo2/.openhands"),
        ).not.toBeInTheDocument();
        expect(
          screen.queryByText("org/repo3/.openhands"),
        ).not.toBeInTheDocument();
        expect(screen.queryByText("user/repo4")).not.toBeInTheDocument();
        expect(
          screen.queryByText("user/TestRepository"),
        ).not.toBeInTheDocument();
        expect(screen.queryByText("org/AnotherRepo")).not.toBeInTheDocument();
      });
    });

    it("should handle special characters in search", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Type in search input with special characters
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, ".openhands");

      // Only repositories with .openhands should be visible
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
      expect(screen.queryByText("user/repo4")).not.toBeInTheDocument();
    });

    it("should maintain accordion functionality with filtered results", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Filter to show only repo2
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, "repo2");

      // Click on the filtered repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for microagents to be fetched
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
      });

      // Check that microagents are displayed
      const microagent1 = screen.getByText("test-microagent-1");
      const microagent2 = screen.getByText("test-microagent-2");

      expect(microagent1).toBeInTheDocument();
      expect(microagent2).toBeInTheDocument();
    });

    it("should handle whitespace in search input", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Type in search input with leading/trailing whitespace
      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });
      await user.type(searchInput, "  repo2  ");

      // repo2 should still be visible (whitespace should be trimmed)
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
    });

    it("should update search results in real-time", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      const searchInput = screen.getByRole("textbox", {
        name: "COMMON$SEARCH_REPOSITORIES",
      });

      // Type "repo" - should show repo2
      await user.type(searchInput, "repo");
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();

      // Add "2" to make it "repo2" - should show only repo2
      await user.type(searchInput, "2");
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();

      // Remove "2" to make it "repo" again - should show repo2
      await user.keyboard("{Backspace}");
      expect(screen.getByTestId("repository-name-tooltip")).toBeInTheDocument();
      expect(screen.queryByText("user/repo1")).not.toBeInTheDocument();
    });
  });

  // Search conversations functionality tests
  describe("Microagent management conversations functionality", () => {
    it("should call useMicroagentManagementConversations API when repository is expanded", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both microagents and conversations to be fetched
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });
    });

    it("should display both microagents and conversations when repository is expanded", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both queries to complete
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that microagents are displayed
      const microagent1 = screen.getByText("test-microagent-1");
      const microagent2 = screen.getByText("test-microagent-2");

      expect(microagent1).toBeInTheDocument();
      expect(microagent2).toBeInTheDocument();

      // Check that conversations are displayed
      const conversation1 = screen.getByText("Test Conversation 1");
      const conversation2 = screen.getByText("Test Conversation 2");

      expect(conversation1).toBeInTheDocument();
      expect(conversation2).toBeInTheDocument();
    });

    it("should show loading state when both microagents and conversations are loading", async () => {
      const user = userEvent.setup();
      mockUseRepositoryMicroagents.mockReturnValue({
        data: undefined,
        isLoading: true,
        isError: false,
      });
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: undefined,
        isLoading: true,
        isError: false,
      });

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Check that loading spinner is displayed
      expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
    });

    it("should hide loading state when both queries complete", async () => {
      const user = userEvent.setup();
      const { container } = renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both queries to complete
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that loading spinner is not displayed
      expect(
        container.querySelector(".animate-indeterminate-spinner"),
      ).toBeNull();
    });

    it("should display microagent file paths for microagents but not for conversations", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both queries to complete
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that microagent file paths are displayed for microagents
      const microagentFilePath1 = screen.getByText(
        ".openhands/microagents/test-microagent-1",
      );
      const microagentFilePath2 = screen.getByText(
        ".openhands/microagents/test-microagent-2",
      );

      expect(microagentFilePath1).toBeInTheDocument();
      expect(microagentFilePath2).toBeInTheDocument();

      // Check that microagent file paths are NOT displayed for conversations
      expect(
        screen.queryByText(".openhands/microagents/Test Conversation 1"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByText(".openhands/microagents/Test Conversation 2"),
      ).not.toBeInTheDocument();
    });

    it("should show learn this repo component when no microagents and no conversations", async () => {
      const user = userEvent.setup();
      mockUseRepositoryMicroagents.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both queries to complete
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that the learn this repo component is displayed
      const learnThisRepo = screen.getByText(
        "MICROAGENT_MANAGEMENT$LEARN_THIS_REPO",
      );
      expect(learnThisRepo).toBeInTheDocument();
    });

    it("should show learn this repo component when only conversations exist but no microagents", async () => {
      const user = userEvent.setup();
      mockUseRepositoryMicroagents.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: [...mockConversations],
        isLoading: false,
        isError: false,
      });

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both queries to complete
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that conversations are displayed
      const conversation1 = screen.getByText("Test Conversation 1");
      const conversation2 = screen.getByText("Test Conversation 2");

      expect(conversation1).toBeInTheDocument();
      expect(conversation2).toBeInTheDocument();

      // Check that learn this repo component is NOT displayed (since we have conversations)
      expect(
        screen.queryByText("MICROAGENT_MANAGEMENT$LEARN_THIS_REPO"),
      ).not.toBeInTheDocument();
    });

    it("should show learn this repo component when only microagents exist but no conversations", async () => {
      const user = userEvent.setup();
      mockUseRepositoryMicroagents.mockReturnValue({
        data: [...mockMicroagents],
        isLoading: false,
        isError: false,
      });
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both queries to complete
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that microagents are displayed
      const microagent1 = screen.getByText("test-microagent-1");
      const microagent2 = screen.getByText("test-microagent-2");

      expect(microagent1).toBeInTheDocument();
      expect(microagent2).toBeInTheDocument();

      // Check that learn this repo component is NOT displayed (since we have microagents)
      expect(
        screen.queryByText("MICROAGENT_MANAGEMENT$LEARN_THIS_REPO"),
      ).not.toBeInTheDocument();
    });

    it("should handle error when fetching conversations", async () => {
      const user = userEvent.setup();
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: undefined,
        isLoading: false,
        isError: true,
      });

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for the error to be handled
      await waitFor(() => {
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that the learn this repo component is displayed (since conversations failed)
      await waitFor(() => {
        expect(
          screen.getByText("MICROAGENT_MANAGEMENT$LEARN_THIS_REPO"),
        ).toBeInTheDocument();
      });

      // Also check that the microagents query was called successfully
      expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
    });

    it("should handle error when fetching microagents but conversations succeed", async () => {
      const user = userEvent.setup();
      mockUseRepositoryMicroagents.mockReturnValue({
        data: undefined,
        isLoading: false,
        isError: true,
      });

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for the error to be handled
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
      });

      // Check that the learn this repo component is displayed (since microagents failed)
      const learnThisRepo = screen.getByText(
        "MICROAGENT_MANAGEMENT$LEARN_THIS_REPO",
      );
      expect(learnThisRepo).toBeInTheDocument();
    });

    it("should call useMicroagentManagementConversations with correct parameters", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for useMicroagentManagementConversations to be called
      await waitFor(() => {
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });
    });

    it("should display conversations with correct information", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for both queries to complete
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that conversations display correct information
      const conversation1 = screen.getByText("Test Conversation 1");
      const conversation2 = screen.getByText("Test Conversation 2");

      expect(conversation1).toBeInTheDocument();
      expect(conversation2).toBeInTheDocument();
    });

    it("should handle multiple repository expansions with conversations", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion
      const repoAccordion1 = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion1);

      // Wait for both queries to be called for first repo
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Check that both microagents and conversations are displayed
      expect(screen.getByText("test-microagent-1")).toBeInTheDocument();
      expect(screen.getByText("test-microagent-2")).toBeInTheDocument();
      expect(screen.getByText("Test Conversation 1")).toBeInTheDocument();
      expect(screen.getByText("Test Conversation 2")).toBeInTheDocument();
    });
  });

  // Add microagent integration tests
  describe("Add microagent functionality", () => {
    beforeEach(() => {
      vi.spyOn(GitService, "getRepositoryBranches").mockResolvedValue({
        branches: [{ name: "main", commit_sha: "abc123", protected: false }],
        has_next_page: false,
        current_page: 1,
        per_page: 30,
        total_count: [{ name: "main", commit_sha: "abc123", protected: false }]
          .length,
      });
    });

    it("should render add microagent button", async () => {
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Check that add microagent buttons are present
      const addButtons = screen.getAllByTestId("add-microagent-button");
      expect(addButtons.length).toBeGreaterThan(0);
    });

    it("should open modal when add button is clicked", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Find and click the first add microagent button
      const addButtons = screen.getAllByTestId("add-microagent-button");
      await user.click(addButtons[0]);

      // Check that the modal is opened
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });
    });

    it("should render modal when Zustand state is set to visible", async () => {
      // Render with modal already visible in Zustand state
      renderWithCustomStore({
        addMicroagentModalVisible: true,
      });

      // Check that modal is rendered
      expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      expect(screen.getByTestId("query-input")).toBeInTheDocument();
      expect(screen.getByTestId("cancel-button")).toBeInTheDocument();
      expect(screen.getByTestId("confirm-button")).toBeInTheDocument();
    });

    it("should display form fields in the modal", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Find and click the first add microagent button
      const addButtons = screen.getAllByTestId("add-microagent-button");
      await user.click(addButtons[0]);

      // Wait for modal to be rendered and check form fields
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Check that form fields are present
      expect(screen.getByTestId("query-input")).toBeInTheDocument();
      expect(screen.getByTestId("cancel-button")).toBeInTheDocument();
      expect(screen.getByTestId("confirm-button")).toBeInTheDocument();
    });

    it("should disable confirm button when query is empty", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Find and click the first add microagent button
      const addButtons = screen.getAllByTestId("add-microagent-button");
      await user.click(addButtons[0]);

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Check that confirm button is disabled when query is empty
      const confirmButton = screen.getByTestId("confirm-button");
      expect(confirmButton).toBeDisabled();
    });

    it("should close modal when cancel button is clicked", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Find and click the first add microagent button
      const addButtons = screen.getAllByTestId("add-microagent-button");
      await user.click(addButtons[0]);

      // Check that the modal is opened
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Click the close button (X icon) - use the first one which should be the modal close button
      const closeButtons = screen.getAllByRole("button", { name: "" });
      const modalCloseButton = closeButtons.find(
        (button) => button.querySelector('svg[height="24"]') !== null,
      );
      await user.click(modalCloseButton!);

      // Check that modal is closed
      await waitFor(() => {
        expect(
          screen.queryByTestId("add-microagent-modal"),
        ).not.toBeInTheDocument();
      });
    });

    it("should enable confirm button when query is entered", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Find and click the first add microagent button
      const addButtons = screen.getAllByTestId("add-microagent-button");
      await user.click(addButtons[0]);

      // Wait for modal to be rendered and branch to be selected
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });
      // Enter query text
      const queryInput = screen.getByTestId("query-input");
      await user.type(queryInput, "Test query");
      // Wait for the confirm button to be enabled after entering query and branch selection
      await waitFor(() => {
        const confirmButton = screen.getByTestId("confirm-button");
        expect(confirmButton).not.toBeDisabled();
      });
    });

    it("should prevent form submission when query is empty", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Find and click the first add microagent button
      const addButtons = screen.getAllByTestId("add-microagent-button");
      await user.click(addButtons[0]);

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Try to submit form with empty query
      const confirmButton = screen.getByTestId("confirm-button");
      await user.click(confirmButton);

      // Check that modal is still open (form submission prevented)
      expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
    });

    it("should trim whitespace from query before submission", async () => {
      const user = userEvent.setup();
      renderMicroagentManagement();

      // Wait for repositories to be loaded and processed
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Wait for repositories to be displayed in the accordion
      await waitFor(() => {
        expect(
          screen.getByTestId("repository-name-tooltip"),
        ).toBeInTheDocument();
      });

      // Find and click the first add microagent button
      const addButtons = screen.getAllByTestId("add-microagent-button");
      await user.click(addButtons[0]);

      // Wait for modal to be rendered and branch to be selected
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });
      // Enter query with whitespace
      const queryInput = screen.getByTestId("query-input");
      await user.type(queryInput, "  Test query with whitespace  ");
      // Wait for the confirm button to be enabled after entering query and branch selection
      await waitFor(() => {
        const confirmButton = screen.getByTestId("confirm-button");
        expect(confirmButton).not.toBeDisabled();
      });
    });
  });

  // MicroagentManagementMain component tests
  describe("MicroagentManagementMain", () => {
    const mockRepositoryMicroagent: RepositoryMicroagent = {
      name: "test-microagent",
      created_at: "2021-10-01T12:00:00Z",
      git_provider: "github",
      path: ".openhands/microagents/test-microagent",
    };

    const mockConversationWithPr: Conversation = {
      conversation_id: "conv-with-pr",
      title: "Test Conversation with PR",
      selected_repository: "user/repo2/.openhands",
      selected_branch: "main",
      git_provider: "github",
      last_updated_at: "2021-10-01T12:00:00Z",
      created_at: "2021-10-01T12:00:00Z",
      status: "RUNNING",
      runtime_status: "STATUS$READY",
      trigger: "microagent_management",
      url: null,
      session_api_key: null,
      pr_number: [123],
    };

    const mockConversationWithoutPr: Conversation = {
      conversation_id: "conv-without-pr",
      title: "Test Conversation without PR",
      selected_repository: "user/repo2/.openhands",
      selected_branch: "main",
      git_provider: "github",
      last_updated_at: "2021-10-01T12:00:00Z",
      created_at: "2021-10-01T12:00:00Z",
      status: "RUNNING",
      runtime_status: "STATUS$READY",
      trigger: "microagent_management",
      url: null,
      session_api_key: null,
      pr_number: [],
    };

    const mockConversationWithNullPr: Conversation = {
      conversation_id: "conv-null-pr",
      title: "Test Conversation with null PR",
      selected_repository: "user/repo2/.openhands",
      selected_branch: "main",
      git_provider: "github",
      last_updated_at: "2021-10-01T12:00:00Z",
      created_at: "2021-10-01T12:00:00Z",
      status: "RUNNING",
      runtime_status: null,
      trigger: "microagent_management",
      url: null,
      session_api_key: null,
      pr_number: null,
    };

    const renderMicroagentManagementMain = (selectedMicroagentItem: any) => {
      // Set the store with the selected microagent item and a repository
      useMicroagentManagementStore.setState({
        selectedMicroagentItem,
        selectedRepository: testRepository,
      });

      return renderWithProviders(<MicroagentManagementMain />);
    };

    it("should render MicroagentManagementDefault when no microagent or conversation is selected", async () => {
      renderMicroagentManagementMain(null);

      // Check that the default component is rendered
      await screen.findByText("MICROAGENT_MANAGEMENT$READY_TO_ADD_MICROAGENT");
      expect(
        screen.getByText(
          "MICROAGENT_MANAGEMENT$OPENHANDS_CAN_LEARN_ABOUT_REPOSITORIES",
        ),
      ).toBeInTheDocument();
    });

    it("should render MicroagentManagementDefault when selectedMicroagentItem is empty object", async () => {
      renderMicroagentManagementMain({});

      // Check that the default component is rendered
      await screen.findByText("MICROAGENT_MANAGEMENT$READY_TO_ADD_MICROAGENT");
      expect(
        screen.getByText(
          "MICROAGENT_MANAGEMENT$OPENHANDS_CAN_LEARN_ABOUT_REPOSITORIES",
        ),
      ).toBeInTheDocument();
    });

    it("should render MicroagentManagementViewMicroagent when microagent is selected", async () => {
      renderMicroagentManagementMain({
        microagent: mockRepositoryMicroagent,
        conversation: null,
      });

      // Check that the microagent view component is rendered
      await screen.findByText("test-microagent");
      expect(
        screen.getByText(".openhands/microagents/test-microagent"),
      ).toBeInTheDocument();
    });

    it("should render MicroagentManagementOpeningPr when conversation is selected with empty pr_number array", async () => {
      renderMicroagentManagementMain({
        microagent: null,
        conversation: mockConversationWithoutPr,
      });

      // Check that the opening PR component is rendered
      await screen.findByText(
        (content) => content === "COMMON$WORKING_ON_IT!",
        { exact: false },
      );
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$WE_ARE_WORKING_ON_IT"),
      ).toBeInTheDocument();
      expect(screen.getAllByTestId("view-conversation-button")).toHaveLength(1);
    });

    it("should render MicroagentManagementOpeningPr when conversation is selected with null pr_number", async () => {
      const conversationWithNullPr = {
        ...mockConversationWithoutPr,
        pr_number: null,
      };
      renderMicroagentManagementMain({
        microagent: null,
        conversation: conversationWithNullPr,
      });

      // Check that the opening PR component is rendered
      await screen.findByText(
        (content) => content === "COMMON$WORKING_ON_IT!",
        { exact: false },
      );
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$WE_ARE_WORKING_ON_IT"),
      ).toBeInTheDocument();
      expect(screen.getAllByTestId("view-conversation-button")).toHaveLength(1);
    });

    it("should render MicroagentManagementReviewPr when conversation is selected with non-empty pr_number array", async () => {
      renderMicroagentManagementMain({
        microagent: null,
        conversation: mockConversationWithPr,
      });

      // Check that the review PR component is rendered
      await screen.findByText("MICROAGENT_MANAGEMENT$YOUR_MICROAGENT_IS_READY");
      expect(screen.getAllByTestId("view-conversation-button")).toHaveLength(2);
    });

    it("should prioritize microagent over conversation when both are present", async () => {
      renderMicroagentManagementMain({
        microagent: mockRepositoryMicroagent,
        conversation: mockConversationWithPr,
      });

      // Should render the microagent view, not the conversation view
      await screen.findByText("test-microagent");
      expect(
        screen.getByText(".openhands/microagents/test-microagent"),
      ).toBeInTheDocument();

      // Should NOT render the review PR component
      expect(
        screen.queryByText("MICROAGENT_MANAGEMENT$YOUR_MICROAGENT_IS_READY"),
      ).not.toBeInTheDocument();
    });

    it("should handle conversation with undefined pr_number", async () => {
      const conversationWithUndefinedPr = {
        ...mockConversationWithoutPr,
      };
      delete conversationWithUndefinedPr.pr_number;

      renderMicroagentManagementMain({
        microagent: null,
        conversation: conversationWithUndefinedPr,
      });

      // Should render the opening PR component (treats undefined as empty array)
      await screen.findByText(
        (content) => content === "COMMON$WORKING_ON_IT!",
        { exact: false },
      );
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$WE_ARE_WORKING_ON_IT"),
      ).toBeInTheDocument();
    });

    it("should handle conversation with multiple PR numbers", async () => {
      const conversationWithMultiplePrs = {
        ...mockConversationWithPr,
        pr_number: [123, 456, 789],
      };

      renderMicroagentManagementMain({
        microagent: null,
        conversation: conversationWithMultiplePrs,
      });

      // Should render the review PR component (non-empty array)
      await screen.findByText("MICROAGENT_MANAGEMENT$YOUR_MICROAGENT_IS_READY");
      expect(screen.getAllByTestId("view-conversation-button")).toHaveLength(2);
    });

    it("should handle conversation with empty string pr_number", async () => {
      const conversationWithEmptyStringPr = {
        ...mockConversationWithoutPr,
        pr_number: "",
      };

      renderMicroagentManagementMain({
        microagent: null,
        conversation: conversationWithEmptyStringPr,
      });

      // Should render the opening PR component (treats empty string as empty array)
      await screen.findByText(
        (content) => content === "COMMON$WORKING_ON_IT!",
        { exact: false },
      );
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$WE_ARE_WORKING_ON_IT"),
      ).toBeInTheDocument();
    });

    it("should handle conversation with zero pr_number", async () => {
      const conversationWithZeroPr = {
        ...mockConversationWithoutPr,
        pr_number: 0,
      };

      renderMicroagentManagementMain({
        microagent: null,
        conversation: conversationWithZeroPr,
      });

      // Should render the opening PR component (treats 0 as falsy)
      await screen.findByText(
        (content) => content === "COMMON$WORKING_ON_IT!",
        { exact: false },
      );
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$WE_ARE_WORKING_ON_IT"),
      ).toBeInTheDocument();
    });

    it("should handle conversation with single PR number as array", async () => {
      const conversationWithSinglePr = {
        ...mockConversationWithPr,
        pr_number: [42],
      };

      renderMicroagentManagementMain({
        microagent: null,
        conversation: conversationWithSinglePr,
      });

      // Should render the review PR component (non-empty array)
      await screen.findByText("MICROAGENT_MANAGEMENT$YOUR_MICROAGENT_IS_READY");
      expect(screen.getAllByTestId("view-conversation-button")).toHaveLength(2);
    });

    it("should handle edge case with null selectedMicroagentItem", async () => {
      renderMicroagentManagementMain(null);

      // Should render the default component
      await screen.findByText("MICROAGENT_MANAGEMENT$READY_TO_ADD_MICROAGENT");
      expect(
        screen.getByText(
          "MICROAGENT_MANAGEMENT$OPENHANDS_CAN_LEARN_ABOUT_REPOSITORIES",
        ),
      ).toBeInTheDocument();
    });

    it("should handle edge case with undefined selectedMicroagentItem", async () => {
      renderMicroagentManagementMain(undefined);

      // Should render the default component
      await screen.findByText("MICROAGENT_MANAGEMENT$READY_TO_ADD_MICROAGENT");
      expect(
        screen.getByText(
          "MICROAGENT_MANAGEMENT$OPENHANDS_CAN_LEARN_ABOUT_REPOSITORIES",
        ),
      ).toBeInTheDocument();
    });

    it("should handle conversation with missing pr_number property", async () => {
      const conversationWithoutPrNumber = {
        conversation_id: "conv-no-pr-number",
        title: "Test Conversation without PR number property",
        selected_repository: "user/repo2/.openhands",
        selected_branch: "main",
        git_provider: "github",
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "RUNNING",
        runtime_status: "STATUS$READY",
        trigger: "microagent_management",
        url: null,
        session_api_key: null,
        // pr_number property is missing
      };

      renderMicroagentManagementMain({
        microagent: null,
        conversation: conversationWithoutPrNumber,
      });

      // Should render the opening PR component (undefined pr_number defaults to empty array)
      await screen.findByText(
        (content) => content === "COMMON$WORKING_ON_IT!",
        { exact: false },
      );
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$WE_ARE_WORKING_ON_IT"),
      ).toBeInTheDocument();
    });

    it("should handle microagent with all required properties", async () => {
      const completeMicroagent: RepositoryMicroagent = {
        name: "complete-microagent",
        created_at: "2021-10-01T12:00:00Z",
        git_provider: "github",
        path: ".openhands/microagents/complete-microagent",
      };

      renderMicroagentManagementMain({
        microagent: completeMicroagent,
        conversation: null,
      });

      // Check that the microagent view component is rendered with complete data
      await screen.findByText("complete-microagent");
      expect(
        screen.getByText(".openhands/microagents/complete-microagent"),
      ).toBeInTheDocument();
    });

    it("should handle conversation with all required properties", async () => {
      const completeConversation: Conversation = {
        conversation_id: "complete-conversation",
        title: "Complete Conversation",
        selected_repository: "user/complete-repo/.openhands",
        selected_branch: "main",
        git_provider: "github",
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "RUNNING",
        runtime_status: "STATUS$READY",
        trigger: "microagent_management",
        url: "https://example.com",
        session_api_key: "test-api-key",
        pr_number: [999],
      };

      renderMicroagentManagementMain({
        microagent: null,
        conversation: completeConversation,
      });

      // Check that the review PR component is rendered with complete data
      await screen.findByText("MICROAGENT_MANAGEMENT$YOUR_MICROAGENT_IS_READY");
      expect(screen.getAllByTestId("view-conversation-button")).toHaveLength(2);
    });
  });

  // Update microagent functionality tests
  describe("Update microagent functionality", () => {
    const mockMicroagentForUpdate: RepositoryMicroagent = {
      name: "update-test-microagent",
      created_at: "2021-10-01T12:00:00Z",
      git_provider: "github",
      path: ".openhands/microagents/update-test-microagent",
    };

    beforeEach(() => {
      vi.spyOn(GitService, "getRepositoryBranches").mockResolvedValue({
        branches: [{ name: "main", commit_sha: "abc123", protected: false }],
        has_next_page: false,
        current_page: 1,
        per_page: 30,
        total_count: [{ name: "main", commit_sha: "abc123", protected: false }]
          .length,
      });
    });

    it("should render update microagent modal when updateMicroagentModalVisible is true", async () => {
      // Render with update modal visible in Zustand state
      renderWithUpdateModal();

      // Check that update modal is rendered
      expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      expect(screen.getByTestId("query-input")).toBeInTheDocument();
      expect(screen.getByTestId("cancel-button")).toBeInTheDocument();
      expect(screen.getByTestId("confirm-button")).toBeInTheDocument();
    });

    it("should display update microagent title when isUpdate is true", async () => {
      // Render with update modal visible and selected microagent
      renderWithUpdateModal();

      // Check that the update title is displayed
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$UPDATE_MICROAGENT"),
      ).toBeInTheDocument();
    });

    it("should populate form fields with existing microagent data when updating", async () => {
      // Render with update modal visible and selected microagent
      renderWithUpdateModal({
        selectedMicroagentItem: {
          microagent: mockMicroagentForUpdate,
          conversation: null,
        },
      });

      // Wait for the content to be loaded and form fields to be populated
      await waitFor(() => {
        const queryInput = screen.getByTestId("query-input");
        expect(queryInput).toHaveValue(
          "Original microagent content for testing updates",
        );
      });
    });

    it("should handle update microagent form submission", async () => {
      const user = userEvent.setup();

      // Render with update modal visible and selected microagent
      renderWithUpdateModal();

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Modify the content
      const queryInput = screen.getByTestId("query-input");
      await user.clear(queryInput);
      await user.type(queryInput, "Updated microagent content");

      // Submit the form
      const confirmButton = screen.getByTestId("confirm-button");
      await user.click(confirmButton);

      // Wait for the modal to be removed after form submission
      await waitFor(() => {
        expect(
          screen.queryByTestId("add-microagent-modal"),
        ).not.toBeInTheDocument();
      });
    });

    it("should close update modal when cancel button is clicked", async () => {
      const user = userEvent.setup();

      // Render with update modal visible
      renderWithUpdateModal();

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Click the cancel button
      const cancelButton = screen.getByTestId("cancel-button");
      await user.click(cancelButton);

      // Check that modal is closed
      await waitFor(() => {
        expect(
          screen.queryByTestId("add-microagent-modal"),
        ).not.toBeInTheDocument();
      });
    });

    it("should close update modal when close button (X) is clicked", async () => {
      const user = userEvent.setup();

      // Render with update modal visible
      renderWithUpdateModal();

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Click the close button (X icon) - use the first one which should be the modal close button
      const closeButtons = screen.getAllByRole("button", { name: "" });
      const modalCloseButton = closeButtons.find(
        (button) =>
          button.querySelector('svg[height="24"]') !== null &&
          !button.hasAttribute("data-testid"),
      );
      await user.click(modalCloseButton!);

      // Check that modal is closed
      await waitFor(() => {
        expect(
          screen.queryByTestId("add-microagent-modal"),
        ).not.toBeInTheDocument();
      });
    });

    it("should handle update modal with empty microagent data", async () => {
      // Render with update modal visible but no microagent data
      renderWithUpdateModal();

      // Check that update modal is still rendered
      expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$UPDATE_MICROAGENT"),
      ).toBeInTheDocument();
    });

    it("should handle update modal with microagent that has no content", async () => {
      const user = userEvent.setup();

      // Mock the content API to return empty content for this test
      vi.spyOn(GitService, "getRepositoryMicroagentContent").mockResolvedValue({
        content: "",
        path: ".openhands/microagents/update-test-microagent",
        git_provider: "github",
        triggers: [],
      });

      // Render with update modal visible and microagent
      renderWithUpdateModal();

      // Wait for the content to be loaded and check that the form field is empty
      await waitFor(() => {
        const queryInput = screen.getByTestId("query-input");
        expect(queryInput).toHaveValue("");
      });
    });

    it("should handle update modal with microagent that has no triggers", async () => {
      const user = userEvent.setup();

      // Mock the content API to return content without triggers for this test
      vi.spyOn(GitService, "getRepositoryMicroagentContent").mockResolvedValue({
        content: "Original microagent content for testing updates",
        path: ".openhands/microagents/update-test-microagent",
        git_provider: "github",
        triggers: [],
      });

      // Render with update modal visible and microagent
      renderWithUpdateModal();

      // Check that the modal is rendered correctly
      expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$UPDATE_MICROAGENT"),
      ).toBeInTheDocument();
    });
  });

  // Learn this repo functionality tests
  describe("Learn this repo functionality", () => {
    it("should display learn this repo trigger when no microagents exist", async () => {
      const user = userEvent.setup();

      // Setup mocks before rendering
      mockUseRepositoryMicroagents.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });

      renderMicroagentManagement();

      // Wait for repositories to be loaded
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      // Find and click on the first repository accordion to expand it
      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      // Wait for microagents and conversations to be fetched
      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Verify the learn this repo trigger is displayed when no microagents exist
      await waitFor(() => {
        expect(
          screen.getByTestId("learn-this-repo-trigger"),
        ).toBeInTheDocument();
      });

      // Verify trigger has correct text content
      expect(screen.getByTestId("learn-this-repo-trigger")).toHaveTextContent(
        "MICROAGENT_MANAGEMENT$LEARN_THIS_REPO",
      );
    });

    it("should trigger learn this repo modal opening when trigger is clicked", async () => {
      const user = userEvent.setup();

      // Setup mocks
      mockUseRepositoryMicroagents.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });

      renderMicroagentManagement();

      // Wait for repositories and expand accordion
      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      await waitFor(() => {
        expect(
          screen.getByTestId("learn-this-repo-trigger"),
        ).toBeInTheDocument();
      });

      // Verify the trigger is clickable and has correct behavior
      const learnThisRepoTrigger = screen.getByTestId(
        "learn-this-repo-trigger",
      );

      // Verify the trigger has the expected text content
      expect(learnThisRepoTrigger).toHaveTextContent(
        "MICROAGENT_MANAGEMENT$LEARN_THIS_REPO",
      );

      // Click the trigger should not throw an error
      await user.click(learnThisRepoTrigger);

      // The trigger should still be present after click (testing that click is handled gracefully)
      expect(learnThisRepoTrigger).toBeInTheDocument();
    });

    it("should show learn this repo trigger only when no microagents or conversations exist", async () => {
      const user = userEvent.setup();

      // Setup mocks with existing microagents (should NOT show trigger)
      mockUseRepositoryMicroagents.mockReturnValue({
        data: [
          {
            name: "test-microagent",
            created_at: "2021-10-01",
            git_provider: "github",
            path: ".openhands/microagents/test",
          },
        ],
        isLoading: false,
        isError: false,
      });
      mockUseMicroagentManagementConversations.mockReturnValue({
        data: [],
        isLoading: false,
        isError: false,
      });

      renderMicroagentManagement();

      await waitFor(() => {
        expect(mockUseGitRepositories).toHaveBeenCalled();
      });

      const repoAccordion = screen.getByTestId("repository-name-tooltip");
      await user.click(repoAccordion);

      await waitFor(() => {
        expect(mockUseRepositoryMicroagents).toHaveBeenCalled();
        expect(mockUseMicroagentManagementConversations).toHaveBeenCalled();
      });

      // Should NOT show the learn this repo trigger when microagents exist
      expect(
        screen.queryByTestId("learn-this-repo-trigger"),
      ).not.toBeInTheDocument();
    });
  });

  // Learn something new button functionality tests
  describe("Learn something new button functionality", () => {
    const mockMicroagentForLearn: RepositoryMicroagent = {
      name: "learn-test-microagent",
      created_at: "2021-10-01T12:00:00Z",
      git_provider: "github",
      path: ".openhands/microagents/learn-test-microagent",
    };

    it("should render learn something new button in microagent view", async () => {
      // Render with selected microagent
      renderWithSelectedMicroagent(mockMicroagentForLearn);

      // Check that the learn something new button is displayed
      expect(
        screen.getByText("COMMON$LEARN_SOMETHING_NEW"),
      ).toBeInTheDocument();
    });

    it("should open update modal when learn something new button is clicked", async () => {
      const user = userEvent.setup();

      // Render with selected microagent
      renderWithSelectedMicroagent(mockMicroagentForLearn);

      // Find and click the learn something new button
      const learnButton = screen.getByText("COMMON$LEARN_SOMETHING_NEW");
      await user.click(learnButton);

      // Check that the update modal is opened
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Check that the update title is displayed
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$UPDATE_MICROAGENT"),
      ).toBeInTheDocument();
    });

    it("should populate form fields with current microagent data when learn button is clicked", async () => {
      const user = userEvent.setup();

      // Mock the content API to return the expected content for this test
      vi.spyOn(GitService, "getRepositoryMicroagentContent").mockResolvedValue({
        content: "Test microagent content for learn functionality",
        path: ".openhands/microagents/learn-test-microagent",
        git_provider: "github",
        triggers: ["learn", "test"],
      });

      // Render with selected microagent
      renderWithSelectedMicroagent(mockMicroagentForLearn);

      // Find and click the learn something new button
      const learnButton = screen.getByText("COMMON$LEARN_SOMETHING_NEW");
      await user.click(learnButton);

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Wait for the content to be loaded and form to be populated
      await waitFor(() => {
        const queryInput = screen.getByTestId("query-input");
        expect(queryInput).toHaveValue(
          "Test microagent content for learn functionality",
        );
      });
    });

    it("should handle learn button click with microagent that has no content", async () => {
      const user = userEvent.setup();

      // Mock the content API to return empty content for this test
      vi.spyOn(GitService, "getRepositoryMicroagentContent").mockResolvedValue({
        content: "",
        path: ".openhands/microagents/learn-test-microagent",
        git_provider: "github",
        triggers: [],
      });

      // Render with selected microagent
      renderWithSelectedMicroagent(mockMicroagentForLearn);

      // Find and click the learn something new button
      const learnButton = screen.getByText("COMMON$LEARN_SOMETHING_NEW");
      await user.click(learnButton);

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Wait for the content to be loaded and check that the form field is empty
      await waitFor(() => {
        const queryInput = screen.getByTestId("query-input");
        expect(queryInput).toHaveValue("");
      });
    });

    it("should handle learn button click with microagent that has no triggers", async () => {
      const user = userEvent.setup();

      // Mock the content API to return content without triggers for this test
      vi.spyOn(GitService, "getRepositoryMicroagentContent").mockResolvedValue({
        content: "Test microagent content for learn functionality",
        path: ".openhands/microagents/learn-test-microagent",
        git_provider: "github",
        triggers: [],
      });

      // Render with selected microagent
      renderWithSelectedMicroagent(mockMicroagentForLearn);

      // Find and click the learn something new button
      const learnButton = screen.getByText("COMMON$LEARN_SOMETHING_NEW");
      await user.click(learnButton);

      // Wait for modal to be rendered
      await waitFor(() => {
        expect(screen.getByTestId("add-microagent-modal")).toBeInTheDocument();
      });

      // Check that the update modal is opened correctly
      expect(
        screen.getByText("MICROAGENT_MANAGEMENT$UPDATE_MICROAGENT"),
      ).toBeInTheDocument();
    });
  });
});
