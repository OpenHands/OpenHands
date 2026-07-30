import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
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

const deviceFlowMocks = vi.hoisted(() => ({
  startDeviceFlow: vi.fn(),
  pollForToken: vi.fn(),
}));

// Partial mock: only the network calls are stubbed so the rest of the module
// (host classification) keeps its production behavior.
vi.mock("#/api/device-flow-client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/device-flow-client")>();
  return {
    ...actual,
    startDeviceFlow: deviceFlowMocks.startDeviceFlow,
    pollForToken: deviceFlowMocks.pollForToken,
  };
});

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

async function selectAgentServer(user = userEvent.setup()) {
  await user.click(screen.getByTestId("add-backend-option-agent-server"));
  return user;
}

beforeEach(() => {
  captureMock = vi.spyOn(telemetry, "trackEvent").mockResolvedValue(undefined);
  window.localStorage.clear();
  getServerInfoMock.mockReset();
  getServerInfoMock.mockResolvedValue({ version: "1.28.0" });
  deviceFlowMocks.startDeviceFlow.mockReset();
  deviceFlowMocks.startDeviceFlow.mockResolvedValue({
    device_code: "device-code",
    user_code: "ABCD-EFGH",
    verification_uri: "https://app.all-hands.dev/device",
    verification_uri_complete:
      "https://app.all-hands.dev/device?user_code=ABCD-EFGH",
    expires_in: 600,
    interval: 5,
  });
  deviceFlowMocks.pollForToken.mockReset();
  deviceFlowMocks.pollForToken.mockImplementation(() => new Promise(() => {}));
  __resetActiveStoreForTests();
});

afterEach(() => {
  captureMock.mockRestore();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AddBackendModal – connection chooser", () => {
  it("renders OpenHands Cloud first with its brand mark", () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(2);
    expect(tabs[0]).toHaveAttribute("data-testid", "add-backend-option-cloud");
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(tabs[1]).toHaveAttribute(
      "data-testid",
      "add-backend-option-agent-server",
    );
    expect(tabs[1]).toHaveAttribute("aria-selected", "false");
    expect(
      within(tabs[0]).getByTestId("add-backend-option-cloud-logo"),
    ).toBeInTheDocument();
    expect(tabs[0]).toHaveTextContent("BACKEND$CLOUD_OPTION_DESCRIPTION");
    expect(tabs[1]).toHaveTextContent(
      "BACKEND$AGENT_SERVER_OPTION_DESCRIPTION",
    );

    expect(screen.getByTestId("add-backend-cloud-panel")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-login-button")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-description")).toHaveTextContent(
      "BACKEND$CHOOSER_DESCRIPTION",
    );
    expect(
      screen.getByTestId("add-backend-deployment-options-link"),
    ).toHaveAttribute(
      "href",
      "https://docs.openhands.dev/overview/introduction",
    );
    // Short inline link so the description reads as one flowing sentence.
    expect(
      screen.getByTestId("add-backend-deployment-options-link"),
    ).toHaveTextContent("CTA$LEARN_MORE");
  });

  it("hides the Advanced host disclosure while authorization is pending", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "open").mockReturnValue({
      closed: false,
      close: vi.fn(),
      location: { href: "" },
    } as unknown as Window);
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    expect(screen.getByTestId("add-backend-advanced-toggle")).toBeVisible();

    await user.click(screen.getByTestId("add-backend-login-button"));

    expect(
      await screen.findByTestId("add-backend-auth-awaiting"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("add-backend-advanced-toggle"),
    ).not.toBeInTheDocument();
  });

  it("shows Local and Remote inside the Agent-server tab", async () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    await selectAgentServer();

    expect(
      screen.getByTestId("add-backend-agent-server-panel"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("add-backend-location-option-local"),
    ).toHaveAttribute("aria-checked", "true");
    expect(
      screen.getByTestId("add-backend-location-option-remote"),
    ).toHaveAttribute("aria-checked", "false");
    expect(
      screen
        .getByTestId("add-backend-location-option-local")
        .querySelector("svg"),
    ).not.toBeNull();
    expect(
      screen
        .getByTestId("add-backend-location-option-remote")
        .querySelector("svg"),
    ).not.toBeNull();
    expect(
      screen.getByTestId("add-backend-local-guidance"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-local-docs-link")).toHaveAttribute(
      "href",
      expect.stringContaining("docs/DEVELOPMENT.md"),
    );
    expect(screen.getByTestId("add-backend-name")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-host")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-api-key")).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-submit")).toBeInTheDocument();
  });

  it("keeps the full setup guidance in a collapsible note", async () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);
    const user = await selectAgentServer();

    expect(
      screen.getByTestId("add-backend-local-guidance-toggle"),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.getByTestId("add-backend-local-guidance-body"),
    ).toHaveAttribute("aria-hidden", "true");

    await user.click(screen.getByTestId("add-backend-local-guidance-toggle"));
    expect(
      screen.getByTestId("add-backend-local-guidance-toggle"),
    ).toHaveAttribute("aria-expanded", "true");
    const localGuidance = screen.getByTestId("add-backend-local-guidance");
    expect(localGuidance).toHaveTextContent("BACKEND$LOCAL_SETUP_DESCRIPTION");
    expect(localGuidance).toHaveTextContent(
      "agent-canvas --backend-only --port 8001",
    );

    await user.click(screen.getByTestId("add-backend-location-option-remote"));
    expect(
      screen.getByTestId("add-backend-remote-guidance-toggle"),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.getByTestId("add-backend-remote-guidance-body"),
    ).toHaveAttribute("aria-hidden", "true");

    await user.click(screen.getByTestId("add-backend-remote-guidance-toggle"));
    const remoteGuidance = screen.getByTestId("add-backend-remote-guidance");
    expect(remoteGuidance).toHaveTextContent(
      "BACKEND$REMOTE_SETUP_DESCRIPTION",
    );
    expect(remoteGuidance).toHaveTextContent(
      "BACKEND$REMOTE_CONNECTION_DESCRIPTION",
    );
  });

  it("starts the Agent-server form with an empty host field", async () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    await selectAgentServer();

    expect(screen.getByTestId("add-backend-host")).toHaveValue("");
  });

  it("disables Connect until name and host are filled (local backend)", async () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);
    const user = await selectAgentServer();

    const submit = screen.getByTestId(
      "add-backend-submit",
    ) as HTMLButtonElement;
    expect(submit).toBeDisabled();

    await user.type(screen.getByTestId("add-backend-name"), "My Server");
    expect(submit).toBeDisabled();

    // Local agent-server connections do not require an API key.
    await user.type(
      screen.getByTestId("add-backend-host"),
      "http://localhost:8000",
    );
    expect(submit).not.toBeDisabled();
  });

  it("allows submitting a local backend with a blank API key", async () => {
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);

    const user = await selectAgentServer();
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

  it("requires an API key for a Remote agent-server", async () => {
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);
    const user = await selectAgentServer();
    await user.click(screen.getByTestId("add-backend-location-option-remote"));

    const submit = screen.getByTestId(
      "add-backend-submit",
    ) as HTMLButtonElement;

    expect(
      screen.getByTestId("add-backend-remote-guidance"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("add-backend-remote-docs-link")).toHaveAttribute(
      "href",
      expect.stringContaining("docs/SELF_HOSTING.md"),
    );

    await user.type(screen.getByTestId("add-backend-name"), "Remote GPU");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "https://agent.example.com",
    );
    expect(submit).toBeDisabled();

    await user.type(screen.getByTestId("add-backend-api-key"), "token");
    expect(submit).not.toBeDisabled();

    await user.click(submit);
    await waitFor(() => expect(onClose).toHaveBeenCalled());

    const stored = JSON.parse(
      window.localStorage.getItem("openhands-backends") ?? "[]",
    );
    expect(
      stored.find((backend: { name: string }) => backend.name === "Remote GPU"),
    ).toMatchObject({
      host: "https://agent.example.com",
      apiKey: "token",
      kind: "local",
    });
  });

  it("saves the backend, switches to it, and closes", async () => {
    const onClose = vi.fn();
    renderWithProviders(<AddBackendModal onClose={onClose} />);

    const user = await selectAgentServer();
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

    const user = await selectAgentServer();
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

    const user = await selectAgentServer();
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

  it("hides advanced host settings until expanded while preserving what was typed", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    // Collapsed: mounted so state survives, but collapsed to zero height and
    // kept out of the tab order.
    expect(screen.getByTestId("add-backend-advanced-panel")).toHaveAttribute(
      "aria-hidden",
      "true",
    );

    await user.click(screen.getByTestId("add-backend-advanced-toggle"));
    expect(screen.getByTestId("add-backend-advanced-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(
      screen.getByTestId("add-backend-advanced-panel"),
    ).not.toHaveAttribute("aria-hidden", "true");

    await user.type(
      screen.getByTestId("add-backend-cloud-host"),
      "https://cloud.example.com",
    );
    await user.click(screen.getByTestId("add-backend-advanced-toggle"));
    await user.click(screen.getByTestId("add-backend-advanced-toggle"));

    expect(screen.getByTestId("add-backend-cloud-host")).toHaveValue(
      "https://cloud.example.com",
    );
  });

  it("renders the cloud login button without a key icon prefix", () => {
    renderWithProviders(<AddBackendModal onClose={vi.fn()} />);

    const loginButton = screen.getByTestId("add-backend-login-button");
    expect(loginButton).not.toHaveClass("w-full");
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
    const user = await selectAgentServer();
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
    const user = await selectAgentServer();

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
