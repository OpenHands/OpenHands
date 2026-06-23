import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useSuggestedTasks } from "../src/hooks/query/use-suggested-tasks";
import { useShouldShowGitFeatures } from "../src/hooks/use-should-show-git-features";

// Mock the dependencies
vi.mock("../src/hooks/use-should-show-git-features");
vi.mock("#/api/suggestions-service/suggestions-service.api", () => ({
  SuggestionsService: {
    getSuggestedTasks: vi.fn().mockResolvedValue([]),
  },
}));

const mockUseShouldShowGitFeatures = vi.mocked(useShouldShowGitFeatures);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe("useSuggestedTasks", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default to disabled
    mockUseShouldShowGitFeatures.mockReturnValue(false);
  });

  it("should be disabled when useShouldShowGitFeatures returns false", () => {
    mockUseShouldShowGitFeatures.mockReturnValue(false);

    const { result } = renderHook(() => useSuggestedTasks(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetching).toBe(false);
  });

  it("should be enabled when useShouldShowGitFeatures returns true", () => {
    mockUseShouldShowGitFeatures.mockReturnValue(true);

    const { result } = renderHook(() => useSuggestedTasks(), {
      wrapper: createWrapper(),
    });

    // When enabled, the query should be loading/fetching
    expect(result.current.isLoading).toBe(true);
  });
});
