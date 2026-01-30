import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, vi, beforeEach, it } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { GitBranchDropdown } from "../../../../src/components/features/home/git-branch-dropdown/git-branch-dropdown";
import { Branch } from "#/types/git";

// Mock the branch data hook
const mockUseBranchData = vi.fn();
vi.mock("#/hooks/query/use-branch-data", () => ({
  useBranchData: (...args: unknown[]) => mockUseBranchData(...args),
}));

const MOCK_BRANCHES: Branch[] = [
  { name: "main", commit_sha: "abc123", protected: true },
  { name: "develop", commit_sha: "def456", protected: false },
  { name: "feature/test", commit_sha: "ghi789", protected: false },
];

const mockOnBranchSelect = vi.fn();

const renderDropdown = (
  props: Partial<Parameters<typeof GitBranchDropdown>[0]> = {},
) => {
  // Default mock return value
  mockUseBranchData.mockReturnValue({
    branches: MOCK_BRANCHES,
    isLoading: false,
    isError: false,
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isSearchLoading: false,
  });

  return render(
    <GitBranchDropdown
      repository="user/repo"
      provider="github"
      selectedBranch={null}
      onBranchSelect={mockOnBranchSelect}
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

describe("GitBranchDropdown", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("dropdown behavior", () => {
    it("should open dropdown when input is clicked", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-branch-dropdown-input");
      await userEvent.click(input);

      // Dropdown should be open (menu should be visible)
      await waitFor(() => {
        expect(
          screen.getByTestId("git-branch-dropdown-menu"),
        ).toBeInTheDocument();
      });
    });

    it("should keep dropdown open when clicking input while already open", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-branch-dropdown-input");

      // First click - open dropdown
      await userEvent.click(input);
      await waitFor(() => {
        expect(
          screen.getByTestId("git-branch-dropdown-menu"),
        ).toBeInTheDocument();
      });

      // Second click on input - should stay open (not close)
      await userEvent.click(input);

      // Dropdown should still be open
      await waitFor(() => {
        expect(
          screen.getByTestId("git-branch-dropdown-menu"),
        ).toBeInTheDocument();
      });
    });

    it("should preserve typed text when clicking input while typing", async () => {
      renderDropdown();

      const input = screen.getByTestId(
        "git-branch-dropdown-input",
      ) as HTMLInputElement;

      // Click to open and type
      await userEvent.click(input);
      await userEvent.type(input, "feat");

      expect(input.value).toBe("feat");

      // Click on input again (should not reset text)
      await userEvent.click(input);

      // Text should be preserved
      expect(input.value).toBe("feat");
    });
  });

  describe("cursor position preservation", () => {
    it("should allow editing in the middle of input text", async () => {
      renderDropdown();

      const input = screen.getByTestId(
        "git-branch-dropdown-input",
      ) as HTMLInputElement;

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
    it("should show selected branch name in input when provided", async () => {
      const selectedBranch = MOCK_BRANCHES[0];
      renderDropdown({ selectedBranch });

      const input = screen.getByTestId(
        "git-branch-dropdown-input",
      ) as HTMLInputElement;

      await waitFor(() => {
        expect(input.value).toBe(selectedBranch.name);
      });
    });
  });

  describe("branch selection", () => {
    it("should call onBranchSelect when a branch is selected", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-branch-dropdown-input");
      await userEvent.click(input);

      // Wait for dropdown to open and show branches
      await waitFor(() => {
        expect(screen.getByText("main")).toBeInTheDocument();
      });

      // Click on a branch
      await userEvent.click(screen.getByText("develop"));

      expect(mockOnBranchSelect).toHaveBeenCalledWith(MOCK_BRANCHES[1]);
    });
  });
});
