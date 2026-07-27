import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, expect, vi, beforeEach, it, afterEach } from "vitest";
import { useUrlSearch } from "#/components/features/home/git-repo-dropdown/use-url-search";
import GitService from "#/api/git-service/git-service.api";

vi.mock("#/api/git-service/git-service.api", () => ({
  default: {
    searchGitRepositories: vi.fn(),
  },
}));

const mockSearchGitRepositories = vi.mocked(GitService.searchGitRepositories);

describe("useUrlSearch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("null/undefined provider guard", () => {
    it("should not call GitService when provider is null", async () => {
      const { result } = renderHook(() =>
        useUrlSearch("https://github.com/owner/repo", null),
      );

      // Wait a tick for the effect to run
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 10);
        });
      });

      expect(mockSearchGitRepositories).not.toHaveBeenCalled();
      expect(result.current.urlSearchResults).toEqual([]);
      expect(result.current.isUrlSearchLoading).toBe(false);
    });

    it("should not call GitService when provider is undefined", async () => {
      const { result } = renderHook(() =>
        useUrlSearch("https://github.com/owner/repo", undefined),
      );

      // Wait a tick for the effect to run
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 10);
        });
      });

      expect(mockSearchGitRepositories).not.toHaveBeenCalled();
      expect(result.current.urlSearchResults).toEqual([]);
      expect(result.current.isUrlSearchLoading).toBe(false);
    });

    it("should clear results when provider becomes null", async () => {
      mockSearchGitRepositories.mockResolvedValue({
        items: [
          { id: "1", full_name: "owner/repo", git_provider: "github", is_public: true },
        ],
        next_page_id: null,
      });

      type TestProps = { inputValue: string; provider: "github" | null };

      const { result, rerender } = renderHook(
        ({ inputValue, provider }: TestProps) => useUrlSearch(inputValue, provider),
        {
          initialProps: {
            inputValue: "https://github.com/owner/repo",
            provider: "github",
          } as TestProps,
        },
      );

      // Wait for initial search to complete
      await waitFor(() => {
        expect(result.current.urlSearchResults).toHaveLength(1);
      });

      // Change provider to null
      rerender({
        inputValue: "https://github.com/owner/repo",
        provider: null,
      });

      // Results should be cleared
      await waitFor(() => {
        expect(result.current.urlSearchResults).toEqual([]);
      });
    });
  });

  describe("URL search behavior", () => {
    it("should call GitService when input is a valid URL and provider is set", async () => {
      mockSearchGitRepositories.mockResolvedValue({
        items: [
          { id: "1", full_name: "owner/repo", git_provider: "github", is_public: true },
        ],
        next_page_id: null,
      });

      const { result } = renderHook(() =>
        useUrlSearch("https://github.com/owner/repo", "github"),
      );

      await waitFor(() => {
        expect(mockSearchGitRepositories).toHaveBeenCalledWith(
          "owner/repo",
          "github",
          3,
        );
      });

      await waitFor(() => {
        expect(result.current.urlSearchResults).toHaveLength(1);
        expect(result.current.urlSearchResults[0].full_name).toBe("owner/repo");
      });
    });

    it("should not call GitService when input is not a URL", async () => {
      const { result } = renderHook(() =>
        useUrlSearch("some search query", "github"),
      );

      // Wait a tick for the effect to run
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 10);
        });
      });

      expect(mockSearchGitRepositories).not.toHaveBeenCalled();
      expect(result.current.urlSearchResults).toEqual([]);
    });

    it("should not call GitService when URL does not match repo pattern", async () => {
      const { result } = renderHook(() =>
        useUrlSearch("https://github.com/", "github"),
      );

      // Wait a tick for the effect to run
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 10);
        });
      });

      expect(mockSearchGitRepositories).not.toHaveBeenCalled();
      expect(result.current.urlSearchResults).toEqual([]);
    });

    it("should handle API errors gracefully", async () => {
      mockSearchGitRepositories.mockRejectedValue(new Error("API Error"));

      const { result } = renderHook(() =>
        useUrlSearch("https://github.com/owner/repo", "github"),
      );

      await waitFor(() => {
        expect(mockSearchGitRepositories).toHaveBeenCalled();
      });

      // Should return empty results on error
      await waitFor(() => {
        expect(result.current.urlSearchResults).toEqual([]);
        expect(result.current.isUrlSearchLoading).toBe(false);
      });
    });

    it("should set loading state correctly during search", async () => {
      let resolveSearch: (value: unknown) => void;
      const searchPromise = new Promise((resolve) => {
        resolveSearch = resolve;
      });

      mockSearchGitRepositories.mockReturnValue(searchPromise as Promise<{
        items: [];
        next_page_id: null;
      }>);

      const { result } = renderHook(() =>
        useUrlSearch("https://github.com/owner/repo", "github"),
      );

      // Should be loading
      await waitFor(() => {
        expect(result.current.isUrlSearchLoading).toBe(true);
      });

      // Resolve the search
      await act(async () => {
        resolveSearch!({ items: [], next_page_id: null });
      });

      // Should no longer be loading
      await waitFor(() => {
        expect(result.current.isUrlSearchLoading).toBe(false);
      });
    });
  });

  describe("clear results on non-URL input", () => {
    it("should clear results when input changes from URL to non-URL", async () => {
      mockSearchGitRepositories.mockResolvedValue({
        items: [
          { id: "1", full_name: "owner/repo", git_provider: "github", is_public: true },
        ],
        next_page_id: null,
      });

      const { result, rerender } = renderHook(
        ({ inputValue, provider }) => useUrlSearch(inputValue, provider),
        {
          initialProps: {
            inputValue: "https://github.com/owner/repo",
            provider: "github" as const,
          },
        },
      );

      // Wait for initial search to complete
      await waitFor(() => {
        expect(result.current.urlSearchResults).toHaveLength(1);
      });

      // Change to non-URL input
      rerender({
        inputValue: "some search",
        provider: "github" as const,
      });

      // Results should be cleared
      await waitFor(() => {
        expect(result.current.urlSearchResults).toEqual([]);
      });
    });
  });
});
