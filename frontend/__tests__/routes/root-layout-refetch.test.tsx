import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import MainApp from "#/routes/root-layout";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";

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
  enable_billing: false,
  hide_llm_settings: false,
  enable_jira: false,
  enable_jira_dc: false,
  enable_linear: false,
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
        app_mode: "saas",
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
          {children}
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
