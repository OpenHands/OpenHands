import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useGitUser } from "#/hooks/query/use-git-user";
import { useLogout } from "#/hooks/mutation/use-logout";
import UserService from "#/api/user-service/user-service.api";
import * as useShouldShowUserFeaturesModule from "#/hooks/use-should-show-user-features";
import * as useConfigModule from "#/hooks/query/use-config";
import { AxiosError } from "axios";

vi.mock("#/hooks/use-should-show-user-features");
vi.mock("#/hooks/query/use-config");
vi.mock("#/hooks/mutation/use-logout");
vi.mock("#/api/user-service/user-service.api");
vi.mock("posthog-js/react", () => ({
  usePostHog: vi.fn(() => ({
    identify: vi.fn(),
  })),
}));

describe("useGitUser", () => {
  let mockLogout: ReturnType<typeof useLogout>;

  beforeEach(() => {
    vi.clearAllMocks();

    mockLogout = {
      mutate: vi.fn(),
      mutateAsync: vi.fn(),
      data: undefined,
      error: null,
      isPending: false,
      isSuccess: false,
      isError: false,
      isIdle: true,
      reset: vi.fn(),
      status: "idle",
    } as unknown as ReturnType<typeof useLogout>;

    vi.mocked(useShouldShowUserFeaturesModule.useShouldShowUserFeatures).mockReturnValue(true);
    vi.mocked(useConfigModule.useConfig).mockReturnValue({
      data: { app_mode: "saas" },
      isLoading: false,
      error: null,
    } as any);
    vi.mocked(useLogout).mockReturnValue(mockLogout);
  });

  const createWrapper = () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };

  it("should call logout when receiving a 401 error", async () => {
    // Mock the user service to throw a 401 error
    const mockError = new AxiosError("Unauthorized", "401", undefined, undefined, {
      status: 401,
      data: { message: "Unauthorized" },
    } as any);

    vi.mocked(UserService.getUser).mockRejectedValue(mockError);

    const { result } = renderHook(() => useGitUser(), {
      wrapper: createWrapper(),
    });

    // Wait for the query to fail (status becomes 'error')
    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });

    // Wait for the useEffect to trigger logout
    await waitFor(() => {
      expect(mockLogout.mutate).toHaveBeenCalled();
    });
  });

  it("should not call logout for non-401 errors", async () => {
    // Mock the user service to throw a 500 error
    const mockError = new AxiosError("Server Error", "500", undefined, undefined, {
      status: 500,
      data: { message: "Internal Server Error" },
    } as any);

    vi.mocked(UserService.getUser).mockRejectedValue(mockError);

    const { result } = renderHook(() => useGitUser(), {
      wrapper: createWrapper(),
    });

    // Wait for the query to fail (status becomes 'error')
    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });

    // Wait a bit to ensure logout is not called
    await waitFor(() => {
      expect(mockLogout.mutate).not.toHaveBeenCalled();
    });
  });
});
