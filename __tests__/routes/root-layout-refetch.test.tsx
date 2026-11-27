import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import MainApp from "#/routes/root-layout";

import { ActiveBackendProvider } from "#/contexts/active-backend-context";

// Hoisted mocks for useIsAuthed and useConfig to allow dynamic control in tests
const { useIsAuthedMock, useConfigMock } = vi.hoisted(() => ({
  useIsAuthedMock: vi.fn(),
  useConfigMock: vi.fn(),
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => useIsAuthedMock(),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));

const DEFAULT_FEATURE_FLAGS = {
  hide_llm_settings: false,
  hide_users_page: false,
};

const RouterStub = createRoutesStub([
  {
    Component: MainApp,
    path: "/",
    children: [
      {
        Component: () => <div data-testid="outlet-content" />,
        path: "/",
      },
      {
        Component: () => <div data-testid="settings-page" />,
        path: "/settings",
      },
    ],
  },
  {
    Component: () => <div data-testid="login-page" />,
    path: "/login",
  },
]);

describe("MainApp - Auth refetch behavior", () => {
  afterEach(async () => {
    // Wait for any pending async operations (e.g., framer-motion LazyMotion async loads)
    // Use act to flush all pending state updates before cleanup
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    cleanup();
    vi.clearAllMocks();
  });

  it("should NOT show loading spinner when auth is refetching for an authenticated user", async () => {
    // Setup: Mock hooks to simulate authenticated user CURRENTLY REFETCHING
    // This is the state when the auth cache is invalidated and refetching
    useIsAuthedMock.mockReturnValue({
      data: true, // Still have cached data showing user is authenticated
      isLoading: false, // Not initial loading
      isFetching: true, // IS refetching - this is the key!
      isError: false,
    });
    useConfigMock.mockReturnValue({
      data: {
        github_client_id: "test-client-id",
        feature_flags: DEFAULT_FEATURE_FLAGS,
      },
      isLoading: false,
    });

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(<RouterStub initialEntries={["/settings"]} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          <ActiveBackendProvider>{children}</ActiveBackendProvider>
        </QueryClientProvider>
      ),
    });

    // BUG: The settings page should still be visible during refetch
    // but the current implementation shows a loading spinner because
    // shouldRedirectToLogin includes isFetchingAuth in its condition
    //
    // This test will FAIL until the bug is fixed.
    // Current behavior: shows full-page loading spinner, redirects to login
    // Expected behavior: shows settings page with root-layout, no redirect

    // Wait a tick for any effects to run
    await waitFor(() => {
      // The root-layout should be present (not replaced by full-page loading spinner)
      const rootLayout = screen.queryByTestId("root-layout");
      // The settings page should remain visible during refetch
      const settingsPage = screen.queryByTestId("settings-page");

      // These assertions describe the EXPECTED behavior (will fail until bug is fixed)
      expect(rootLayout).toBeInTheDocument();
      expect(settingsPage).toBeInTheDocument();
    });
  });
});
