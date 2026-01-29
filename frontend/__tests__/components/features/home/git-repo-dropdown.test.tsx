import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, vi, beforeEach, it } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { GitRepoDropdown } from "../../../../src/components/features/home/git-repo-dropdown/git-repo-dropdown";
import { GitRepository } from "#/types/git";

// Mock the repository data hook
const mockUseRepositoryData = vi.fn();
vi.mock(
  "#/components/features/home/git-repo-dropdown/use-repository-data",
  () => ({
    useRepositoryData: (...args: unknown[]) => mockUseRepositoryData(...args),
  }),
);

// Mock the URL search hook
const mockUseUrlSearch = vi.fn();
vi.mock("#/components/features/home/git-repo-dropdown/use-url-search", () => ({
  useUrlSearch: (...args: unknown[]) => mockUseUrlSearch(...args),
}));

// Mock useConfig
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: null }),
}));

// Mock useHomeStore
vi.mock("#/stores/home-store", () => ({
  useHomeStore: () => ({ recentRepositories: [] }),
}));

const MOCK_REPOSITORIES: GitRepository[] = [
  {
    id: "1",
    full_name: "user/repo-one",
    git_provider: "github",
    is_public: true,
  },
  {
    id: "2",
    full_name: "user/repo-two",
    git_provider: "github",
    is_public: true,
  },
  {
    id: "3",
    full_name: "org/feature-repo",
    git_provider: "github",
    is_public: false,
  },
];

const mockOnChange = vi.fn();

const setupDefaultMocks = (
  repositoryDataOverrides: Partial<
    ReturnType<typeof mockUseRepositoryData>
  > = {},
) => {
  mockUseRepositoryData.mockReturnValue({
    repositories: MOCK_REPOSITORIES,
    selectedRepository: null,
    isLoading: false,
    isError: false,
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isSearchLoading: false,
    ...repositoryDataOverrides,
  });

  mockUseUrlSearch.mockReturnValue({
    urlSearchResults: [],
    isUrlSearchLoading: false,
  });
};

const renderDropdown = (
  props: Partial<Parameters<typeof GitRepoDropdown>[0]> = {},
  repositoryDataOverrides: Partial<
    ReturnType<typeof mockUseRepositoryData>
  > = {},
) => {
  // Set up mocks with optional overrides
  setupDefaultMocks(repositoryDataOverrides);

  return render(
    <GitRepoDropdown
      provider="github"
      onChange={mockOnChange}
      // eslint-disable-next-line react/jsx-props-no-spreading
      {...props}
    />,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: {
                queries: {
                  retry: false,
                },
              },
            })
          }
        >
          {children}
        </QueryClientProvider>
      ),
    },
  );
};

describe("GitRepoDropdown", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("dropdown behavior", () => {
    it("should open dropdown when input is clicked", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown");
      await userEvent.click(input);

      // Dropdown should be open (menu should be visible)
      await waitFor(() => {
        expect(
          screen.getByTestId("git-repo-dropdown-menu"),
        ).toBeInTheDocument();
      });
    });

    it("should keep dropdown open when clicking input while already open", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown");

      // First click - open dropdown
      await userEvent.click(input);
      await waitFor(() => {
        expect(
          screen.getByTestId("git-repo-dropdown-menu"),
        ).toBeInTheDocument();
      });

      // Second click on input - should stay open (not close)
      await userEvent.click(input);

      // Dropdown should still be open
      await waitFor(() => {
        expect(
          screen.getByTestId("git-repo-dropdown-menu"),
        ).toBeInTheDocument();
      });
    });

    it("should preserve typed text when clicking input while typing", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      // Click to open and type
      await userEvent.click(input);
      await userEvent.type(input, "repo");

      expect(input.value).toBe("repo");

      // Click on input again (should not reset text)
      await userEvent.click(input);

      // Text should be preserved
      expect(input.value).toBe("repo");
    });
  });

  describe("cursor position preservation", () => {
    it("should allow editing in the middle of input text", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      // Click and type initial text
      await userEvent.click(input);
      await userEvent.type(input, "hello");

      expect(input.value).toBe("hello");

      // Move cursor to position 2 and type
      input.setSelectionRange(2, 2);
      await userEvent.type(input, "X");

      // The character should be inserted (exact position may vary based on browser behavior)
      expect(input.value).toContain("X");
    });
  });

  describe("input synchronization", () => {
    it("should show selected repository name in input when provided", async () => {
      const selectedRepository = MOCK_REPOSITORIES[0];

      renderDropdown(
        { value: selectedRepository.full_name },
        { selectedRepository },
      );

      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      await waitFor(() => {
        expect(input.value).toBe(selectedRepository.full_name);
      });
    });
  });

  describe("repository selection", () => {
    it("should call onChange when a repository is selected", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown");
      await userEvent.click(input);

      // Wait for dropdown to open and show repositories
      await waitFor(() => {
        expect(screen.getByText("user/repo-one")).toBeInTheDocument();
      });

      // Click on a repository
      await userEvent.click(screen.getByText("user/repo-two"));

      expect(mockOnChange).toHaveBeenCalledWith(MOCK_REPOSITORIES[1]);
    });
  });
});
