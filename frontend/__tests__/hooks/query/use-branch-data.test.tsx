import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useBranchData } from "#/hooks/query/use-branch-data";
import { Branch } from "#/types/git";

const mockFetchNextListPage = vi.fn();
const mockFetchNextSearchPage = vi.fn();

const listBranches: Branch[] = [
  { name: "main", commit_sha: "abc", protected: true },
  { name: "develop", commit_sha: "def", protected: false },
];

const searchBranches: Branch[] = [
  { name: "feature/a", commit_sha: "111", protected: false },
  { name: "feature/b", commit_sha: "222", protected: false },
];

vi.mock("#/hooks/query/use-repository-branches", () => ({
  useRepositoryBranchesPaginated: vi.fn(() => ({
    data: { pages: [{ items: listBranches }] },
    fetchNextPage: mockFetchNextListPage,
    hasNextPage: true,
    isLoading: false,
    isFetchingNextPage: false,
    isError: false,
  })),
}));

vi.mock("#/hooks/query/use-search-branches", () => ({
  useSearchBranches: vi.fn(
    (_repository: string | null, query: string) => {
      if (!query) {
        return {
          data: [],
          fetchNextPage: mockFetchNextSearchPage,
          hasNextPage: false,
          isLoading: false,
          isFetchingNextPage: false,
          isError: false,
        };
      }

      return {
        data: searchBranches,
        fetchNextPage: mockFetchNextSearchPage,
        hasNextPage: true,
        isLoading: true,
        isFetchingNextPage: true,
        isError: false,
      };
    },
  ),
}));

describe("useBranchData", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("uses list pagination when search input matches selected branch", () => {
    const selectedBranch = listBranches[0];

    const { result } = renderHook(() =>
      useBranchData(
        "user/repo",
        "github",
        null,
        "main",
        "main",
        selectedBranch,
      ),
    );

    expect(result.current.branches).toEqual(listBranches);
    expect(result.current.fetchNextPage).toBe(mockFetchNextListPage);
    expect(result.current.hasNextPage).toBe(true);
    expect(result.current.isFetchingNextPage).toBe(false);
    expect(result.current.isLoading).toBe(false);
  });

  it("uses search pagination and loading state while searching", () => {
    const { result } = renderHook(() =>
      useBranchData("user/repo", "github", null, "feature", "feature", null),
    );

    expect(result.current.branches).toEqual(searchBranches);
    expect(result.current.fetchNextPage).toBe(mockFetchNextSearchPage);
    expect(result.current.hasNextPage).toBe(true);
    expect(result.current.isFetchingNextPage).toBe(true);
    expect(result.current.isLoading).toBe(true);
    expect(result.current.isSearchLoading).toBe(true);
  });
});
