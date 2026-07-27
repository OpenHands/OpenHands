import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";
import {
  BACKEND_HEALTH_STORAGE_KEY,
  MAX_CONSECUTIVE_FAILURES,
} from "#/api/backend-registry/health-storage";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { __resetHealthStoreForTests } from "#/api/backend-registry/health-store";
import {
  ActiveBackendProvider,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import { ManageBackendsModal } from "#/components/features/backends/manage-backends-modal";
import { BackendVersion } from "#/components/features/backends/backend-version";
import { BackendRow } from "#/components/features/backends/backend-row";
import { type Backend } from "#/api/backend-registry/types";
import { CLOUD_BACKEND_LOGGED_OUT_ERROR } from "#/hooks/query/use-backends-health";
import {
  getCloudOrganizations,
  getCloudOrganizationMe,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";
import * as telemetry from "#/services/telemetry";

const deviceFlowMocks = vi.hoisted(() => ({
  startDeviceFlow: vi.fn(),
  pollForToken: vi.fn(),
}));

const getServerInfoMock = vi.fn().mockResolvedValue({ version: "1.28.0" });
const getSettingsMock = vi.fn().mockResolvedValue({});

vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(function ServerClientMock() {
    return { getServerInfo: getServerInfoMock };
  }),
  SettingsClient: vi.fn(function SettingsClientMock() {
    return { getSettings: getSettingsMock };
  }),
}));

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCloudOrganizations: vi.fn(),
  getCloudOrganizationMe: vi.fn(),
  getCurrentCloudApiKey: vi.fn(),
}));

// Partial mock: only the device-flow network calls are stubbed. The pure
// host-classifier `isOpenHandsCloudHost` (which the backend form uses to infer
// a backend's kind) is inherited from the real module so classification stays
// faithful to production.
vi.mock("#/api/device-flow-client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/device-flow-client")>();
  return {
    ...actual,
    startDeviceFlow: deviceFlowMocks.startDeviceFlow,
    pollForToken: deviceFlowMocks.pollForToken,
  };
});

let captureMock: ReturnType<typeof vi.spyOn>;

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: { user_consents_to_analytics: true, email: "user@example.com" },
  }),
}));

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>{ui}</ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

function TestSeed({
  onMount,
  children,
}: {
  onMount: (ctx: ReturnType<typeof useActiveBackendContext>) => void;
  children: React.ReactNode;
}) {
  const ctx = useActiveBackendContext();
  React.useEffect(() => {
    onMount(ctx);
  }, []);
  return children as React.ReactElement;
}

beforeEach(() => {
  captureMock = vi.spyOn(telemetry, "trackEvent").mockResolvedValue(undefined);
  vi.spyOn(telemetry, "isTelemetryEnabled").mockReturnValue(true);
  window.localStorage.clear();
  getServerInfoMock.mockReset();
  getServerInfoMock.mockResolvedValue({ version: "1.28.0" });
  getSettingsMock.mockReset();
  getSettingsMock.mockResolvedValue({});
  vi.mocked(getCloudOrganizations).mockReset();
  vi.mocked(getCloudOrganizations).mockResolvedValue({
    items: [],
    currentOrgId: null,
  });
  vi.mocked(getCloudOrganizationMe).mockReset();
  vi.mocked(getCloudOrganizationMe).mockResolvedValue({
    orgId: "",
    userId: "",
    role: null,
  });
  vi.mocked(getCurrentCloudApiKey).mockReset();
  vi.mocked(getCurrentCloudApiKey).mockResolvedValue({
    orgId: null,
    isLegacyKey: true,
  });
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
  __resetHealthStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_LOCK_TO_CLOUD__;
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
});

describe("ManageBackendsModal", () => {
  it("renders a status dot in each row", async () => {
    renderWithProviders(<ManageBackendsModal onClose={vi.fn()} />);

    expect(
      await screen.findByTestId("manage-backends-modal"),
    ).toBeInTheDocument();

    // The seeded default Local backend row is present and has a status
    // indicator alongside its name + host.
    const dots = await screen.findAllByTestId("backend-status-dot");
    expect(dots.length).toBeGreaterThanOrEqual(1);
  });

  it("shows invalid API key status when the backend auth probe returns 401", async () => {
    getSettingsMock.mockRejectedValue(
      Object.assign(new Error("Unauthorized"), {
        name: "HttpError",
        status: 401,
      }),
    );

    renderWithProviders(<ManageBackendsModal onClose={vi.fn()} />);

    const row = await screen.findByTestId("manage-backends-row-Local");
    await waitFor(() =>
      expect(
        screen.getByTestId("manage-backends-status-Local"),
      ).toHaveTextContent("BACKEND$STATUS_DISCONNECTED_CHECK_API_KEY"),
    );
    expect(
      row.querySelector('[data-testid="backend-status-dot"]'),
    ).toHaveAttribute("data-status", "disconnected");
  });

  it("shows disconnected for a reachable backend below the compatible version floor", async () => {
    getServerInfoMock.mockResolvedValue({ version: "1.27.1" });

    renderWithProviders(<ManageBackendsModal onClose={vi.fn()} />);

    expect(
      await screen.findByTestId("manage-backends-version-Local"),
    ).toBeInTheDocument();
    expect(getServerInfoMock).toHaveBeenCalled();
    await waitFor(() =>
      expect(
        screen.getByTestId("manage-backends-status-Local"),
      ).toHaveTextContent("ONBOARDING$BACKEND_STATUS_DISCONNECTED"),
    );
    expect(
      screen.getByTestId("manage-backends-status-detail-Local"),
    ).toHaveTextContent("Agent Canvas requires agent-server 1.28.0 or newer");
  });

  it("closes when the header close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithProviders(<ManageBackendsModal onClose={onClose} />);

    await user.click(await screen.findByTestId("close-manage-backends-modal"));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("opens the add-backend form when '+ Add backend' is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ManageBackendsModal onClose={vi.fn()} />);

    await user.click(screen.getByTestId("manage-backends-add"));

    expect(await screen.findByTestId("add-backend-modal")).toBeInTheDocument();
  });

  it("re-checks a disabled backend on open and clears stale persisted health when it recovers", async () => {
    window.localStorage.setItem(
      BACKEND_HEALTH_STORAGE_KEY,
      JSON.stringify({
        [SEEDED_DEFAULT_BACKEND_ID]: {
          consecutiveFailures: MAX_CONSECUTIVE_FAILURES,
          lastError: "Network Error",
          lastFailureAt: Date.now(),
          disabled: true,
        },
      }),
    );
    __resetHealthStoreForTests();

    renderWithProviders(<ManageBackendsModal onClose={vi.fn()} />);

    await waitFor(() => {
      expect(
        window.localStorage.getItem(BACKEND_HEALTH_STORAGE_KEY),
      ).toBeNull();
    });
  });

  it("opens an edit form pre-filled with the row's backend, and persists changes via updateBackend", async () => {
    // These tests exercise edit-form behavior, not lock behavior; isolate
    // them from a local .env that sets VITE_LOCK_TO_CLOUD and would hide the
    // pencil button this test depends on.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;

    const user = userEvent.setup();

    let backendId = "";
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          backendId = ctx.addBackend({
            name: "Acme Local",
            host: "http://localhost:9000",
            apiKey: "old-key",
            kind: "local",
          }).id;
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} />
      </TestSeed>,
    );

    await user.click(
      await screen.findByTestId("manage-backends-edit-Acme Local"),
    );

    await screen.findByTestId("edit-backend-modal");
    const nameInput = screen.getByTestId(
      "edit-backend-name",
    ) as HTMLInputElement;
    expect(nameInput.value).toBe("Acme Local");

    // Update the host and save.
    const hostInput = screen.getByTestId(
      "edit-backend-host",
    ) as HTMLInputElement;
    await user.clear(hostInput);
    await user.type(hostInput, "http://localhost:9999");

    await user.click(screen.getByTestId("edit-backend-submit"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("edit-backend-modal"),
      ).not.toBeInTheDocument();
    });

    // The list now reflects the new host and the original id is preserved.
    const row = screen.getByTestId("manage-backends-row-Acme Local");
    expect(row.textContent).toContain("http://localhost:9999");
    expect(backendId).not.toBe("");
  });

  it("preserves kind:cloud when renaming a cloud backend on a custom domain", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;

    const user = userEvent.setup();

    let backendId = "";
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          backendId = ctx.addBackend({
            name: "OHE Prod",
            host: "https://app.company.com",
            apiKey: "sk-oh-test",
            kind: "cloud",
          }).id;
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} />
      </TestSeed>,
    );

    await user.click(
      await screen.findByTestId("manage-backends-edit-OHE Prod"),
    );
    await screen.findByTestId("edit-backend-modal");

    // Rename only -- host and kind are untouched
    const nameInput = screen.getByTestId(
      "edit-backend-name",
    ) as HTMLInputElement;
    await user.clear(nameInput);
    await user.type(nameInput, "OHE Prod Renamed");

    await user.click(screen.getByTestId("edit-backend-submit"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("edit-backend-modal"),
      ).not.toBeInTheDocument();
    });

    const stored = JSON.parse(
      window.localStorage.getItem("openhands-backends") ?? "[]",
    );
    const updated = stored.find((b: { id: string }) => b.id === backendId);
    expect(updated).toMatchObject({
      name: "OHE Prod Renamed",
      kind: "cloud",
    });
  });

  it("closes the edit form when the header close button is clicked", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;

    const user = userEvent.setup();

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "Acme Local",
            host: "http://localhost:9000",
            apiKey: "old-key",
            kind: "local",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} />
      </TestSeed>,
    );

    await user.click(
      await screen.findByTestId("manage-backends-edit-Acme Local"),
    );
    await screen.findByTestId("edit-backend-modal");

    await user.click(screen.getByTestId("edit-backend-close"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("edit-backend-modal"),
      ).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("manage-backends-modal")).toBeInTheDocument();
  });

  it("shows each cloud backend's connected organization so same-named backends are distinguishable", async () => {
    // Two cloud backends share the same name and host; each API key is bound
    // to a different org, which is the only thing that tells them apart.
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: "org-acme", name: "Acme Inc" },
        { id: "org-beta", name: "Beta Co" },
      ],
      currentOrgId: "org-acme",
    });
    vi.mocked(getCurrentCloudApiKey).mockImplementation(async (backend) => ({
      orgId: backend?.apiKey === "key-acme" ? "org-acme" : "org-beta",
      isLegacyKey: false,
    }));

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "Production",
            host: "https://app.all-hands.dev",
            apiKey: "key-acme",
            kind: "cloud",
          });
          ctx.addBackend({
            name: "Production",
            host: "https://app.all-hands.dev",
            apiKey: "key-beta",
            kind: "cloud",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} />
      </TestSeed>,
    );

    // Each otherwise-identical row now surfaces its own bound organization.
    expect(await screen.findByText("Acme Inc")).toBeInTheDocument();
    expect(screen.getByText("Beta Co")).toBeInTheDocument();
  });

  it("labels a cloud backend's personal workspace instead of showing its raw org name", async () => {
    const personalOrgId = "0b93b5f2-5396-49f2-8d98-61f906184270";
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [{ id: personalOrgId, name: `user_${personalOrgId}_org` }],
      currentOrgId: personalOrgId,
    });
    vi.mocked(getCurrentCloudApiKey).mockResolvedValue({
      orgId: personalOrgId,
      isLegacyKey: false,
    });
    // /me reports user_id === org_id, so the bound org is the user's personal
    // workspace and must render the friendly label, not the backend-side name.
    vi.mocked(getCloudOrganizationMe).mockResolvedValue({
      orgId: personalOrgId,
      userId: personalOrgId,
      role: null,
    });

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "Production",
            host: "https://app.all-hands.dev",
            apiKey: "bearer-key",
            kind: "cloud",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} />
      </TestSeed>,
    );

    expect(
      await screen.findByText("BACKEND$PERSONAL_WORKSPACE"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(`user_${personalOrgId}_org`),
    ).not.toBeInTheDocument();
  });

  it("does not render an organization line for a local backend", async () => {
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "Acme Local",
            host: "http://localhost:9000",
            apiKey: "k",
            kind: "local",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} />
      </TestSeed>,
    );

    expect(
      await screen.findByTestId("manage-backends-row-Acme Local"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("manage-backends-org-Acme Local"),
    ).not.toBeInTheDocument();
  });

  it("captures backend_added with source manage_backends_modal when adding from here", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ManageBackendsModal onClose={vi.fn()} />);

    await user.click(await screen.findByTestId("manage-backends-add"));
    await screen.findByTestId("add-backend-modal");

    await user.type(screen.getByTestId("add-backend-name"), "Local Extra");
    await user.type(
      screen.getByTestId("add-backend-host"),
      "http://localhost:8000",
    );
    await user.type(screen.getByTestId("add-backend-api-key"), "sk-local");
    await user.click(screen.getByTestId("add-backend-submit"));

    await waitFor(() =>
      expect(captureMock).toHaveBeenCalledWith(
        "backend_added",
        expect.objectContaining({ source: "manage_backends_modal" }),
      ),
    );
  });

  it("does not capture backend_added when editing an existing backend", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "Acme Local",
            host: "http://localhost:9000",
            apiKey: "old-key",
            kind: "local",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} />
      </TestSeed>,
    );

    await user.click(
      await screen.findByTestId("manage-backends-edit-Acme Local"),
    );
    await screen.findByTestId("edit-backend-modal");
    const hostInput = screen.getByTestId("edit-backend-host");
    await user.clear(hostInput);
    await user.type(hostInput, "http://localhost:9999");
    await user.click(screen.getByTestId("edit-backend-submit"));

    await waitFor(() =>
      expect(
        screen.queryByTestId("edit-backend-modal"),
      ).not.toBeInTheDocument(),
    );
    expect(captureMock).not.toHaveBeenCalledWith(
      "backend_added",
      expect.anything(),
    );
  });
  it("uses Cloud reconnect recovery controls when locked to Cloud", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    const user = userEvent.setup();
    vi.spyOn(window, "open").mockReturnValue({
      closed: false,
      close: vi.fn(),
      location: { href: "" },
    } as unknown as Window);

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "OpenHands Cloud",
            host: "https://app.all-hands.dev",
            apiKey: "expired-token",
            kind: "cloud",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} recoveryMode />
      </TestSeed>,
    );

    expect(
      await screen.findByRole("heading", {
        name: "BACKEND$RECONNECT_CLOUD_TITLE",
      }),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("manage-backends-add")).not.toBeInTheDocument();

    const reconnectButton = await screen.findByTestId(
      "manage-backends-reconnect-cloud-login-button",
    );
    expect(reconnectButton).toHaveTextContent("BACKEND$RECONNECT_CLOUD");

    await user.click(reconnectButton);
    expect(deviceFlowMocks.startDeviceFlow).toHaveBeenCalledWith(
      "https://app.all-hands.dev",
    );
  });

  it("updates the locked Cloud backend token when reconnect succeeds", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    deviceFlowMocks.pollForToken.mockResolvedValue({
      access_token: "fresh-cloud-token",
      token_type: "Bearer",
    });
    const user = userEvent.setup();
    vi.spyOn(window, "open").mockReturnValue({
      closed: false,
      close: vi.fn(),
      location: { href: "" },
    } as unknown as Window);

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "OpenHands Cloud",
            host: "https://app.all-hands.dev",
            apiKey: "expired-token",
            kind: "cloud",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} recoveryMode />
      </TestSeed>,
    );

    await user.click(
      await screen.findByTestId(
        "manage-backends-reconnect-cloud-login-button",
      ),
    );

    await waitFor(() => {
      const stored = JSON.parse(
        window.localStorage.getItem("openhands-backends") ?? "[]",
      );
      expect(stored).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: "OpenHands Cloud",
            apiKey: "fresh-cloud-token",
          }),
        ]),
      );
    });
    expect(
      screen.queryByTestId("manage-backends-reconnect-cloud-auth-modal"),
    ).not.toBeInTheDocument();
  });

  it("keeps locked Cloud recovery open when reconnect authorization fails", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    deviceFlowMocks.pollForToken.mockRejectedValue(
      new Error("Authorization request was denied."),
    );
    const user = userEvent.setup();
    vi.spyOn(window, "open").mockReturnValue({
      closed: false,
      close: vi.fn(),
      location: { href: "" },
    } as unknown as Window);

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "OpenHands Cloud",
            host: "https://app.all-hands.dev",
            apiKey: "expired-token",
            kind: "cloud",
          });
        }}
      >
        <ManageBackendsModal onClose={vi.fn()} recoveryMode />
      </TestSeed>,
    );

    await user.click(
      await screen.findByTestId(
        "manage-backends-reconnect-cloud-login-button",
      ),
    );

    expect(
      await screen.findByTestId("manage-backends-reconnect-cloud-auth-error"),
    ).toHaveTextContent("Authorization request was denied.");
    expect(screen.getByTestId("manage-backends-modal")).toBeInTheDocument();
  });
});

function renderInQueryClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

const cloudBackend: Backend = {
  id: "backend-cloud",
  name: "Acme Cloud",
  host: "https://app.acme.com",
  apiKey: "sk-test",
  kind: "cloud",
};

// Focused coverage for the two components extracted out of this file. The
// ManageBackendsModal suite above already exercises them through the modal;
// these assert behavior unique to each extracted component, rendered without
// ActiveBackendProvider so no seeded-backend health probe interferes.
describe("BackendVersion", () => {
  it("renders no version badge and skips the probe for a non-local backend", () => {
    // The version probe is gated on `kind === "local"`, so a cloud backend
    // never fetches or shows a version.
    renderInQueryClient(<BackendVersion backend={cloudBackend} />);

    expect(
      screen.queryByTestId(`manage-backends-version-${cloudBackend.name}`),
    ).not.toBeInTheDocument();
    expect(getServerInfoMock).not.toHaveBeenCalled();
  });
});

describe("BackendRow", () => {
  it("disables row selection when the backend is not connected", () => {
    renderInQueryClient(
      <ul>
        <BackendRow
          backend={cloudBackend}
          health={undefined}
          onSelect={vi.fn()}
          onEdit={vi.fn()}
          onRemove={vi.fn()}
        />
      </ul>,
    );

    const row = screen.getByTestId(`manage-backends-row-${cloudBackend.name}`);
    const selectButton = within(row)
      .getByText(cloudBackend.name)
      .closest("button");
    expect(selectButton).toBeDisabled();
  });

  it("shows a logged-out cloud status with a log back in button", () => {
    const onLogin = vi.fn();

    renderInQueryClient(
      <ul>
        <BackendRow
          backend={cloudBackend}
          health={{
            isConnected: false,
            consecutiveFailures: 1,
            lastError: CLOUD_BACKEND_LOGGED_OUT_ERROR,
            disabled: false,
          }}
          onSelect={vi.fn()}
          onEdit={vi.fn()}
          onRemove={vi.fn()}
          onLogin={onLogin}
        />
      </ul>,
    );

    expect(
      screen.getByTestId(`manage-backends-status-${cloudBackend.name}`),
    ).toHaveTextContent("BACKEND$LOGGED_OUT");
    expect(
      screen.queryByTestId(
        `manage-backends-status-detail-${cloudBackend.name}`,
      ),
    ).not.toBeInTheDocument();
    const loginButton = screen.getByTestId(
      `manage-backends-login-${cloudBackend.id}-login-button`,
    );
    expect(loginButton).toHaveAccessibleName("BACKEND$LOG_BACK_IN");
    expect(loginButton).not.toHaveTextContent("BACKEND$LOG_BACK_IN");
    expect(loginButton.querySelector("svg")).toBeInTheDocument();
    expect(loginButton).toHaveClass("hover:bg-interactive-hover");
    expect(loginButton).not.toHaveClass("border");
    expect(loginButton).not.toHaveClass("bg-primary");
  });

  it("shows device authorization in a modal instead of expanding the row", async () => {
    const user = userEvent.setup();
    const popup = {
      closed: false,
      close: vi.fn(),
      location: { href: "" },
    } as unknown as Window;
    vi.spyOn(window, "open").mockReturnValue(popup);

    renderInQueryClient(
      <ul>
        <BackendRow
          backend={cloudBackend}
          health={{
            isConnected: false,
            consecutiveFailures: 1,
            lastError: CLOUD_BACKEND_LOGGED_OUT_ERROR,
            disabled: false,
          }}
          onSelect={vi.fn()}
          onEdit={vi.fn()}
          onRemove={vi.fn()}
          onLogin={vi.fn()}
        />
      </ul>,
    );

    const row = screen.getByTestId(`manage-backends-row-${cloudBackend.name}`);
    await user.click(
      screen.getByTestId(
        `manage-backends-login-${cloudBackend.id}-login-button`,
      ),
    );

    const modal = await screen.findByTestId(
      `manage-backends-login-${cloudBackend.id}-auth-modal`,
    );
    expect(
      await within(modal).findByTestId(
        `manage-backends-login-${cloudBackend.id}-auth-awaiting`,
      ),
    ).toBeInTheDocument();
    expect(
      within(row).queryByTestId(
        `manage-backends-login-${cloudBackend.id}-auth-awaiting`,
      ),
    ).not.toBeInTheDocument();
    expect(deviceFlowMocks.startDeviceFlow).toHaveBeenCalledWith(
      cloudBackend.host,
    );
  });

  it("renders edit and remove buttons when the deployment is not locked to a cloud host", async () => {
    // The repo's .env may set VITE_LOCK_TO_CLOUD; clear it explicitly so this
    // test exercises the unlocked code path regardless of local env state.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "");
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;

    const user = userEvent.setup();
    const onEdit = vi.fn();
    const onRemove = vi.fn();

    renderInQueryClient(
      <ul>
        <BackendRow
          backend={cloudBackend}
          health={{
            isConnected: true,
            consecutiveFailures: 0,
            lastError: null,
            disabled: false,
          }}
          onSelect={vi.fn()}
          onEdit={onEdit}
          onRemove={onRemove}
        />
      </ul>,
    );

    const row = screen.getByTestId(`manage-backends-row-${cloudBackend.name}`);
    const editButton = within(row).getByTestId(
      `manage-backends-edit-${cloudBackend.name}`,
    );
    const removeButton = within(row).getByTestId(
      `manage-backends-remove-${cloudBackend.name}`,
    );

    expect(editButton).toHaveAccessibleName("BACKEND$EDIT");
    expect(removeButton).toHaveAccessibleName("BACKEND$REMOVE");

    await user.click(editButton);
    await user.click(removeButton);
    expect(onEdit).toHaveBeenCalledTimes(1);
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it("hides edit and remove buttons when locked to a cloud host via VITE_LOCK_TO_CLOUD", () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://cloud.example.com");
    const onEdit = vi.fn();
    const onRemove = vi.fn();

    renderInQueryClient(
      <ul>
        <BackendRow
          backend={cloudBackend}
          health={{
            isConnected: true,
            consecutiveFailures: 0,
            lastError: null,
            disabled: false,
          }}
          onSelect={vi.fn()}
          onEdit={onEdit}
          onRemove={onRemove}
        />
      </ul>,
    );

    const row = screen.getByTestId(`manage-backends-row-${cloudBackend.name}`);
    expect(
      within(row).queryByTestId(`manage-backends-edit-${cloudBackend.name}`),
    ).not.toBeInTheDocument();
    expect(
      within(row).queryByTestId(`manage-backends-remove-${cloudBackend.name}`),
    ).not.toBeInTheDocument();
  });

  it("hides edit and remove buttons when locked to a cloud host via window.__AGENT_CANVAS_LOCK_TO_CLOUD__", () => {
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_LOCK_TO_CLOUD__ = "https://cloud.example.com";

    renderInQueryClient(
      <ul>
        <BackendRow
          backend={cloudBackend}
          health={{
            isConnected: true,
            consecutiveFailures: 0,
            lastError: null,
            disabled: false,
          }}
          onSelect={vi.fn()}
          onEdit={vi.fn()}
          onRemove={vi.fn()}
        />
      </ul>,
    );

    const row = screen.getByTestId(`manage-backends-row-${cloudBackend.name}`);
    expect(
      within(row).queryByTestId(`manage-backends-edit-${cloudBackend.name}`),
    ).not.toBeInTheDocument();
    expect(
      within(row).queryByTestId(`manage-backends-remove-${cloudBackend.name}`),
    ).not.toBeInTheDocument();
    // Row identity (name + host) is still rendered so the user can see the
    // locked backend is selected, just not mutate it.
    expect(within(row).getByText(cloudBackend.name)).toBeInTheDocument();
    expect(within(row).getByText(cloudBackend.host)).toBeInTheDocument();
  });
});
