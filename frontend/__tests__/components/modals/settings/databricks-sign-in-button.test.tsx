import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DatabricksSignInButton } from "#/components/shared/modals/settings/databricks-sign-in-button";

vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  return {
    ...actual,
    useTranslation: () => ({
      // Default-value fallback mirrors the i18next behaviour used by the
      // component so we assert against the English copy without maintaining a
      // duplicate translation map here.
      t: (key: string, opts?: { defaultValue?: string }) =>
        opts?.defaultValue ?? key,
    }),
  };
});

// Prevent custom-toast-handlers from pulling in the full i18n init chain.
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

const statusMock = vi.fn();
const logoutMock = vi.fn();

vi.mock(
  "#/api/databricks-auth-service/databricks-auth-service.api",
  () => ({
    default: {
      INITIATE_URL: "/auth/databricks/initiate",
      status: (...args: unknown[]) => statusMock(...args),
      logout: (...args: unknown[]) => logoutMock(...args),
    },
  }),
);

function renderWithClient(ui: React.ReactElement) {
  // Fresh client per test so cached status from a previous render cannot
  // leak across cases.
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
  });
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
  );
}

describe("DatabricksSignInButton", () => {
  const originalLocation = window.location;

  beforeEach(() => {
    statusMock.mockReset();
    logoutMock.mockReset();
    // jsdom's window.location is not assignable by default — replace it with
    // a mutable stub so we can observe navigation from handleSignIn.
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: { ...originalLocation, href: "http://localhost/" },
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: originalLocation,
    });
  });

  it("renders nothing when the provider is not active", () => {
    renderWithClient(<DatabricksSignInButton isActive={false} />);
    expect(
      screen.queryByTestId("databricks-sign-in-button"),
    ).not.toBeInTheDocument();
    expect(statusMock).not.toHaveBeenCalled();
  });

  it("renders nothing when U2M is not configured on the backend", async () => {
    statusMock.mockResolvedValue({
      configured: false,
      authenticated: false,
      host: null,
    });

    renderWithClient(<DatabricksSignInButton isActive />);

    await waitFor(() => expect(statusMock).toHaveBeenCalled());
    expect(
      screen.queryByTestId("databricks-sign-in-button"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("databricks-signed-in"),
    ).not.toBeInTheDocument();
  });

  it("shows the Sign-in CTA when configured but not authenticated", async () => {
    statusMock.mockResolvedValue({
      configured: true,
      authenticated: false,
      host: null,
    });

    renderWithClient(<DatabricksSignInButton isActive />);

    const btn = await screen.findByTestId("databricks-sign-in-button");
    expect(btn).toHaveTextContent(/sign in with databricks/i);
  });

  it("opens a new tab to the initiate URL on click (env-var path)", async () => {
    statusMock.mockResolvedValue({
      configured: true,
      authenticated: false,
      host: null,
    });

    const openSpy = vi.spyOn(window, "open").mockReturnValue(null);

    const user = userEvent.setup();
    renderWithClient(<DatabricksSignInButton isActive />);

    const btn = await screen.findByTestId("databricks-sign-in-button");
    await user.click(btn);

    // Env-var path: no client_id → window.open with the initiate URL in a new tab.
    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining("/auth/databricks/initiate"),
      "_blank",
    );
    openSpy.mockRestore();
  });

  it("shows the signed-in host and Sign-out button when authenticated", async () => {
    statusMock.mockResolvedValue({
      configured: true,
      authenticated: true,
      host: "https://adb-123.cloud.databricks.com",
    });

    renderWithClient(<DatabricksSignInButton isActive />);

    expect(
      await screen.findByTestId("databricks-signed-in"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("https://adb-123.cloud.databricks.com"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("databricks-sign-out-button"),
    ).toHaveTextContent(/sign out of databricks/i);
    // No Sign-in CTA while signed in.
    expect(
      screen.queryByTestId("databricks-sign-in-button"),
    ).not.toBeInTheDocument();
  });

  it("calls logout when the Sign-out button is clicked", async () => {
    statusMock.mockResolvedValue({
      configured: true,
      authenticated: true,
      host: "https://adb-123.cloud.databricks.com",
    });
    logoutMock.mockResolvedValue(undefined);

    const user = userEvent.setup();
    renderWithClient(<DatabricksSignInButton isActive />);

    const signOut = await screen.findByTestId("databricks-sign-out-button");
    await user.click(signOut);

    await waitFor(() => expect(logoutMock).toHaveBeenCalledTimes(1));
    // Location should not change for sign-out (it is an XHR POST, not a
    // full-page redirect).
    expect(window.location.href).toBe("http://localhost/");
  });
});
