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
  it("should clear prior results when an HTTPS URL does not match repo pattern", async () => {
    mockSearchGitRepositories.mockResolvedValue({
      items: [
        {
          id: "1",
          full_name: "owner/repo",
          git_provider: "github",
          is_public: true,
        },
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

    await waitFor(() => {
      expect(result.current.urlSearchResults).toHaveLength(1);
    });

    rerender({
      inputValue: "https://example.com/",
      provider: "github" as const,
    });

    await waitFor(() => {
      expect(result.current.urlSearchResults).toEqual([]);
    });

    // Only the initial search for owner/repo should have triggered a call;
    // the non-matching HTTPS URL must not issue a second request.
    expect(mockSearchGitRepositories).toHaveBeenCalledTimes(1);
  });

  describe("superseded requests", () => {
    type SearchResult = Awaited<
      ReturnType<typeof GitService.searchGitRepositories>
    >;

    const repo = (fullName: string) => ({
      id: "1",
      full_name: fullName,
      git_provider: "github" as const,
      is_public: true,
    });

    const pending = () => {
      const resolvers: ((value: SearchResult) => void)[] = [];
      const rejecters: ((reason: unknown) => void)[] = [];
      mockSearchGitRepositories.mockImplementation(
        () =>
          new Promise<SearchResult>((resolve, reject) => {
            resolvers.push(resolve);
            rejecters.push(reject);
          }),
      );
      return { resolvers, rejecters };
    };

    it("should not list a result for input the user has already cleared", async () => {
      const { resolvers } = pending();

      type TestProps = { inputValue: string };
      const { result, rerender } = renderHook(
        ({ inputValue }: TestProps) => useUrlSearch(inputValue, "github"),
        {
          initialProps: {
            inputValue: "https://github.com/owner/repo-119",
          } as TestProps,
        },
      );

      await waitFor(() => {
        expect(mockSearchGitRepositories).toHaveBeenCalledTimes(1);
      });

      // The user clears the field while that request is still in flight.
      rerender({ inputValue: "" });

      // Only now does the earlier request come back.
      await act(async () => {
        resolvers[0]({ items: [repo("owner/repo-119")], next_page_id: null });
      });

      expect(result.current.urlSearchResults).toEqual([]);
      expect(result.current.isUrlSearchLoading).toBe(false);
    });

    it("should stop the spinner when the field is cleared mid-request", async () => {
      pending();

      type TestProps = { inputValue: string };
      const { result, rerender } = renderHook(
        ({ inputValue }: TestProps) => useUrlSearch(inputValue, "github"),
        {
          initialProps: {
            inputValue: "https://github.com/owner/repo-119",
          } as TestProps,
        },
      );

      await waitFor(() => {
        expect(result.current.isUrlSearchLoading).toBe(true);
      });

      rerender({ inputValue: "" });

      await waitFor(() => {
        expect(result.current.isUrlSearchLoading).toBe(false);
      });
    });

    it("should stop the spinner when the provider is cleared mid-request", async () => {
      pending();

      type TestProps = { provider: "github" | null };
      const { result, rerender } = renderHook(
        ({ provider }: TestProps) =>
          useUrlSearch("https://github.com/owner/repo-119", provider),
        { initialProps: { provider: "github" } as TestProps },
      );

      await waitFor(() => {
        expect(result.current.isUrlSearchLoading).toBe(true);
      });

      rerender({ provider: null });

      await waitFor(() => {
        expect(result.current.isUrlSearchLoading).toBe(false);
      });
    });

    it("should ignore an earlier request that resolves after a newer one started", async () => {
      const { resolvers } = pending();

      type TestProps = { inputValue: string };
      const { result, rerender } = renderHook(
        ({ inputValue }: TestProps) => useUrlSearch(inputValue, "github"),
        {
          initialProps: {
            inputValue: "https://github.com/owner/repo-119",
          } as TestProps,
        },
      );

      await waitFor(() => {
        expect(mockSearchGitRepositories).toHaveBeenCalledTimes(1);
      });

      rerender({ inputValue: "https://github.com/owner/repo-200" });

      await waitFor(() => {
        expect(mockSearchGitRepositories).toHaveBeenCalledTimes(2);
      });

      // The first request finishes late, after the second one is already out.
      await act(async () => {
        resolvers[0]({ items: [repo("owner/repo-119")], next_page_id: null });
      });

      expect(result.current.urlSearchResults).toEqual([]);
      // The second request is still pending, so the spinner stays on.
      expect(result.current.isUrlSearchLoading).toBe(true);
    });

    it("should not let an earlier failure clear a newer request's results", async () => {
      const { resolvers, rejecters } = pending();

      type TestProps = { inputValue: string };
      const { result, rerender } = renderHook(
        ({ inputValue }: TestProps) => useUrlSearch(inputValue, "github"),
        {
          initialProps: {
            inputValue: "https://github.com/owner/repo-119",
          } as TestProps,
        },
      );

      await waitFor(() => {
        expect(mockSearchGitRepositories).toHaveBeenCalledTimes(1);
      });

      rerender({ inputValue: "https://github.com/owner/repo-200" });

      await waitFor(() => {
        expect(mockSearchGitRepositories).toHaveBeenCalledTimes(2);
      });

      await act(async () => {
        resolvers[1]({ items: [repo("owner/repo-200")], next_page_id: null });
      });

      expect(result.current.urlSearchResults).toEqual([repo("owner/repo-200")]);

      // The superseded request now fails. It must not wipe the current results.
      await act(async () => {
        rejecters[0](new Error("superseded request failed"));
      });

      expect(result.current.urlSearchResults).toEqual([repo("owner/repo-200")]);
    });
  });
});
