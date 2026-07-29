import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockInstance } from "vitest";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { AddBackendModal } from "#/components/features/backends/add-backend-modal";
import * as telemetry from "#/services/telemetry";

const getServerInfoMock = vi.hoisted(() => vi.fn());

vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(function ServerClientMock() {
    return {
      getServerInfo: getServerInfoMock,
    };
  }),
}));

let captureMock: MockInstance<typeof telemetry.trackEvent>;

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: { user_consents_to_analytics: true, email: "user@example.com" },
  }),
}));

function renderWithProviders(
  ui: React.ReactElement,
  navigation?: NavigationContextValue,
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        {navigation ? (
          <NavigationProvider value={navigation}>{ui}</NavigationProvider>
        ) : (
          ui
        )}
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  captureMock = vi.spyOn(telemetry, "trackEvent").mockResolvedValue(undefined);
  window.localStorage.clear();
  getServerInfoMock.mockReset();
  getServerInfoMock.mockResolvedValue({ version: "1.28.0" });
  __resetActiveStoreForTests();
});

afterEach(() => {
  captureMock.mockRestore();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AddBackendModal – two-column layout", () => {
  it("renders a two-column layout with manual and cloud sections", () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    expect(screen.getByTestId("add-backend-name")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-host")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-host-helper")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-api-key")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-submit")).toBeInTheDocument();

    expect(screen.getByTestId("add-backend-cloud-title")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-login-button")).toBeInTheDocument();
    expect(
      screen.getByTestId("add-backend-advanced-toggle"),
    ).toBeInTheDocument();
  });

  it("starts with an empty host field (no prefilled value)", () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    expect(screen.getByTestId("add-backend-host")).toHaveValue("");
  });

  it("disables Connect until name and host are filled (local backend)", async () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    const submit = screen.getByTestId(
      "add-backend-submit",
    ) as HTMLButtonElement;
    expect(submit).toBeDisabled();

    const user = userEvent.setup();
    await user.type(screen.getByTestId("add-backend-name"), "My Server");
    expect(submit).toBeDisabled();

    // A localhost host infers "local" kind → no API key required
    await user.type(
      screen.getByTestId("add-backend-host"),
      "http://localhost:8000",
    );
    expect(submit).not.toBeDisabled();
  });

  it("allows submitting a local backend with a blank API key", async () => {
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);

    const user = userEvent.setup();
    await user.type(screen.getByTestId("add-backend-name"), "Local Extra");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "http://127.0.0.1:18002",
    );

    await user.click(screen.getByTestId("add-backend-submit"));

    await waitFor(() => expect(onClose).toHaveBeenCalled());

    const stored = JSON.parse(
      window.localStorage.getItem("openhands-backends") ?? "[]",
    );
    const added = stored.find(
      (b: { name: string }) => b.name === "Local Extra",
    );
    expect(added).toMatchObject({
      name: "Local Extra",
      host: "http://127.0.0.1:18002",
      apiKey: "",
      kind: "local",
    });
  });

  it("requires API key when host infers cloud kind", async () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    const submit = screen.getByTestId(
      "add-backend-submit",
    ) as HTMLButtonElement;
    const user = userEvent.setup();

    await user.type(screen.getByTestId("add-backend-name"), "Cloud");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "https://app.openhands.dev",
    );
    // Cloud host without API key → submit should be disabled
    expect(submit).toBeDisabled();

    await user.type(screen.getByTestId("add-backend-api-key"), "token");
    expect(submit).not.toBeDisabled();
  });

  it("saves the backend, switches to it, and closes", async () => {
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);

    const user = userEvent.setup();
    await user.type(screen.getByTestId("add-backend-name"), "Local 1");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "http://localhost:9000",
    );
    await user.type(screen.getByTestId("add-backend-api-key"), "k");

    await user.click(screen.getByTestId("add-backend-submit"));

    await waitFor(() => expect(onClose).toHaveBeenCalled());

    const stored = JSON.parse(
      window.localStorage.getItem("openhands-backends") ?? "[]",
    );
    expect(stored).toHaveLength(2);
    const added = stored.find((b: { name: string }) => b.name === "Local 1");
    expect(added).toMatchObject({
      name: "Local 1",
      host: "http://localhost:9000",
      apiKey: "k",
      kind: "local",
    });

    // Active selection must point at the newly added backend.
    const active = JSON.parse(
      window.localStorage.getItem("openhands-active-backend") ?? "null",
    );
    expect(active).toEqual({ backendId: added.id, orgId: null });
  });

  it("keeps the modal open and shows a connection error when the local backend probe fails", async () => {
    getServerInfoMock.mockRejectedValueOnce(new Error("Failed to fetch"));
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);

    const user = userEvent.setup();
    await user.type(screen.getByTestId("add-backend-name"), "GPU Tunnel");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "https://127.0.0.1:8000",
    );
    await user.type(screen.getByTestId("add-backend-api-key"), "session-key");
    await user.click(screen.getByTestId("add-backend-submit"));

    expect(await screen.findByTestId("add-backend-error")).toHaveTextContent(
      "BACKEND$CONNECTION_TEST_FAILED",
    );
    expect(screen.getByTestId("add-backend-error")).toHaveTextContent(
      "Disconnected",
    );
    expect(onClose).not.toHaveBeenCalled();
  });

  it("keeps the modal open when the local backend is below the compatible version floor", async () => {
    getServerInfoMock.mockResolvedValueOnce({ version: "1.27.1" });
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);

    const user = userEvent.setup();
    await user.type(screen.getByTestId("add-backend-name"), "Old Tunnel");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "https://127.0.0.1:8000",
    );
    await user.type(screen.getByTestId("add-backend-api-key"), "session-key");
    await user.click(screen.getByTestId("add-backend-submit"));

    expect(await screen.findByTestId("add-backend-error")).toHaveTextContent(
      "Agent Canvas requires agent-server 1.28.0 or newer",
    );
    expect(onClose).not.toHaveBeenCalled();
  });

  it("closes when the header close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);

    await user.click(screen.getByTestId("add-backend-close"));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("keeps advanced host settings in the layout when collapsed", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    expect(screen.getByTestId("add-backend-cloud-host")).toBeInTheDocument();

    await user.click(screen.getByTestId("add-backend-advanced-toggle"));

    expect(screen.getByTestId("add-backend-cloud-host")).toBeInTheDocument();
  });

  it("renders the cloud login button without a key icon prefix", () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    const loginButton = screen.getByTestId("add-backend-login-button");
    expect(loginButton.textContent?.trim()).not.toMatch(/^🔑/);
    expect(loginButton.textContent).not.toContain("🔑");
  });
});

// @spec BM-002 — adding a backend auto-switches the active selection, so a
// backend-scoped detail page is now stale; the user must land on the section
// list rather than the previous backend's detail page.
describe("AddBackendModal – redirect after adding a backend", () => {
  function renderOnPath(currentPath: string) {
    const navigate = vi.fn();
    const navigation: NavigationContextValue = {
      currentPath,
      conversationId: null,
      isNavigating: false,
      navigate,
    };
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />, navigation);
    return { navigate };
  }

  async function addLocalBackend() {
    const user = userEvent.setup();
    await user.type(screen.getByTestId("add-backend-name"), "Local Extra");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "http://127.0.0.1:18002",
    );
    await user.click(screen.getByTestId("add-backend-submit"));
    await waitFor(() => {
      const stored = JSON.parse(
        window.localStorage.getItem("openhands-backends") ?? "[]",
      );
      expect(
        stored.some((b: { name: string }) => b.name === "Local Extra"),
      ).toBe(true);
    });
  }

  it.each([
    { path: "/automations/auto-1", expected: "/automations" },
    { path: "/conversations/abc", expected: "/conversations" },
  ])(
    "redirects to the section list when adding from $path",
    async ({ path, expected }) => {
      // Arrange
      const { navigate } = renderOnPath(path);

      // Act
      await addLocalBackend();

      // Assert
      expect(navigate).toHaveBeenCalledWith(expected);
    },
  );

  it("does not redirect when adding from a section list page", async () => {
    // Arrange
    const { navigate } = renderOnPath("/automations");

    // Act
    await addLocalBackend();

    // Assert
    expect(navigate).not.toHaveBeenCalled();
  });
});

describe("AddBackendModal – analytics", () => {
  it("captures backend_added once with manual connection metadata", async () => {
    // Arrange
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);
    const user = userEvent.setup();

    // Act — connect a local backend through the manual form
    await user.type(screen.getByTestId("add-backend-name"), "Local Extra");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "http://localhost:8000",
    );
    await user.type(screen.getByTestId("add-backend-api-key"), "sk-local");
    await user.click(screen.getByTestId("add-backend-submit"));

    // Assert — emitted exactly once with coarse, non-sensitive properties
    await waitFor(() =>
      expect(captureMock).toHaveBeenCalledWith(
        "backend_added",
        expect.objectContaining({
          backend_kind: "local",
          connection_method: "manual",
          has_api_key: true,
          source: "add_backend_modal",
        }),
      ),
    );
    const backendAddedCalls = captureMock.mock.calls.filter(
      ([event]) => event === "backend_added",
    );
    expect(backendAddedCalls).toHaveLength(1);
  });
});
