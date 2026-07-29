import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRoutesStub, MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import {
  ActiveBackendProvider,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import { BackendSelector } from "#/components/features/backends/backend-selector";
import {
  __resetEnvironmentSwitchOverlayForTests,
  EnvironmentSwitchOverlay,
} from "#/components/features/backends/environment-switch-overlay";
import { ENVIRONMENT_SWITCH_SETACTIVE_DELAY_MS } from "#/components/features/backends/environment-switch-store";

import {
  ServerClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import {
  getCloudOrganizations,
  getCloudOrganizationMe,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCloudOrganizations: vi.fn(),
  getCloudOrganizationMe: vi.fn(),
  getCurrentCloudApiKey: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(),
  SettingsClient: vi.fn(),
}));

// Shared seed configs reused across tests.
const SEED_LOCAL_1 = {
  name: "Local 1",
  host: "http://localhost:9000",
  apiKey: "k",
  kind: "local" as const,
};

const SEED_CLOUD_PRODUCTION = {
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud" as const,
};

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{ui}</ActiveBackendProvider>
      </QueryClientProvider>
    </MemoryRouter>,
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return children as React.ReactElement;
}

async function openDropdown() {
  const user = userEvent.setup();
  const wrapper = screen.getByTestId("backend-selector");
  // BackendSelector renders Dropdown with `openOnHover` while the trigger is
  // visible — clicking the toggle button would open via hover and then
  // immediately close via the click handler. Hovering the wrapper alone is
  // enough to surface the menu.
  await user.hover(wrapper);
  return user;
}

beforeEach(() => {
  window.localStorage.clear();
  vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
  vi.stubEnv("VITE_SESSION_API_KEY", "session-key");
  __resetActiveStoreForTests();
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
  // Default to the legacy-key fallback so existing assertions about
  // multiple orgs being visible still hold. Tests that exercise
  // org-filter behavior override this with an explicit orgId.
  vi.mocked(getCurrentCloudApiKey).mockReset();
  vi.mocked(getCurrentCloudApiKey).mockResolvedValue({
    orgId: null,
    isLegacyKey: true,
  });
  // Default the local-server health probe to "connected" so the
  // existing label-focused tests aren't surprised by a red indicator;
  // tests that assert on the disconnected state override this.
  vi.mocked(ServerClient).mockReset();
  vi.mocked(ServerClient).mockImplementation(function ServerClientMock() {
    return {
      getServerInfo: vi.fn().mockResolvedValue({ version: "1.28.0" }),
    } as unknown as ServerClient;
  });
  vi.mocked(SettingsClient).mockReset();
  vi.mocked(SettingsClient).mockImplementation(function SettingsClientMock() {
    return {
      getSettings: vi.fn().mockResolvedValue({}),
    } as unknown as SettingsClient;
  });
});

afterEach(async () => {
  // When a test selects a dropdown option, BackendSelector's onChange
  // calls `triggerEnvironmentSwitch` and then awaits a real-time
  // `setTimeout(..., ENVIRONMENT_SWITCH_SETACTIVE_DELAY_MS)` before
  // running `setActive`. `userEvent.click` does NOT await that async
  // handler, so the trailing `setActive` can land AFTER the test ends,
  // re-polluting `localStorage` during a later test. The body's
  // `data-environment-switching` attribute is set while the switch is
  // in flight; wait it out before clearing storage so the in-flight
  // `setActive` writes BEFORE we wipe state.
  if (document.body.hasAttribute("data-environment-switching")) {
    await new Promise((resolve) => {
      setTimeout(resolve, ENVIRONMENT_SWITCH_SETACTIVE_DELAY_MS + 20);
    });
  }
  window.localStorage.clear();
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
  __resetEnvironmentSwitchOverlayForTests();
});

describe("BackendSelector", () => {
  it("shows the seeded default Local backend's name by default", () => {
    renderWithProviders(<BackendSelector />);
    const wrapper = screen.getByTestId("backend-selector");
    const input = wrapper.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("Local");
  });

  it("lists all registered backends in the dropdown", async () => {
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend(SEED_LOCAL_1);
          ctx.addBackend({
            name: "Production",
            host: "https://app.all-hands.dev",
            apiKey: "b",
            kind: "cloud",
          });
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    await openDropdown();

    // The seeded default ("Local") is rendered as a normal row alongside
    // the user-added backends.
    expect(screen.getByText("Local")).toBeInTheDocument();
    expect(screen.getByText("Local 1")).toBeInTheDocument();
    expect(screen.getByText("Production")).toBeInTheDocument();
  });

  it("expands a cloud backend into one row per org and records the org locally without calling cloud /switch", async () => {
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: "org-personal", name: "Personal" },
        { id: "org-2", name: "Acme Inc" },
      ],
      currentOrgId: "org-personal",
    });

    let cloudId = "";
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          cloudId = ctx.addBackend(SEED_CLOUD_PRODUCTION).id;
          // Add a second backend so auto-switch lands here, leaving the
          // cloud backend unselected (the dropdown only expands org rows
          // for non-active cloud backends).
          ctx.addBackend(SEED_LOCAL_1);
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    const user = await openDropdown();

    await waitFor(() => {
      expect(screen.getByText("Production – Personal")).toBeInTheDocument();
    });
    expect(screen.getByText("Production – Acme Inc")).toBeInTheDocument();

    await user.click(screen.getByText("Production – Acme Inc"));

    // Selecting an org row updates the active selection locally only —
    // it must never trigger the cloud-mutating /switch call that used to
    // leak the local-UI choice into the cloud UI's `current_org_id`.
    await waitFor(() => {
      const stored = JSON.parse(
        window.localStorage.getItem("openhands-active-backend") ?? "null",
      );
      expect(stored).toEqual({ backendId: cloudId, orgId: "org-2" });
    });
  });

  it("filters each cloud backend's org rows to the org its API key is bound to", async () => {
    // Both backends point at the same host; the user belongs to all three
    // orgs, but each API key is scoped to a different one. The selector
    // must show one row per backend, each labeled with its key's own org.
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: "org-personal", name: "Personal" },
        { id: "org-acme", name: "Acme Inc" },
        { id: "org-beta", name: "Beta Co" },
      ],
      currentOrgId: "org-personal",
    });
    vi.mocked(getCurrentCloudApiKey).mockImplementation(async (backend) => ({
      orgId: backend?.apiKey === "key-personal" ? "org-personal" : "org-acme",
      isLegacyKey: false,
    }));

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "ProdPersonal",
            host: "https://app.all-hands.dev",
            apiKey: "key-personal",
            kind: "cloud",
          });
          ctx.addBackend({
            name: "ProdAcme",
            host: "https://app.all-hands.dev",
            apiKey: "key-acme",
            kind: "cloud",
          });
          // Land on a local backend so both cloud backends are unselected
          // and their org rows render in the dropdown.
          ctx.addBackend(SEED_LOCAL_1);
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    await openDropdown();

    await waitFor(() => {
      expect(screen.getByText("ProdPersonal – Personal")).toBeInTheDocument();
    });
    expect(screen.getByText("ProdAcme – Acme Inc")).toBeInTheDocument();
    // Inaccessible orgs must not appear under either backend.
    expect(
      screen.queryByText("ProdPersonal – Acme Inc"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("ProdAcme – Personal")).not.toBeInTheDocument();
    expect(screen.queryByText(/Beta Co/)).not.toBeInTheDocument();
  });

  it("shows every organization as a separate row for cookie-auth cloud backends", async () => {
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: "org-personal", name: "Personal" },
        { id: "org-acme", name: "Acme Inc" },
      ],
      currentOrgId: "org-acme",
    });
    vi.mocked(getCurrentCloudApiKey).mockResolvedValue({
      orgId: "org-personal",
      isLegacyKey: false,
    });

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend({
            name: "OpenHands Cloud",
            host: "https://app.all-hands.dev",
            apiKey: "",
            kind: "cloud",
            authMode: "cookie",
          });
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    await waitFor(() => {
      const wrapper = screen.getByTestId("backend-selector");
      const input = wrapper.querySelector("input") as HTMLInputElement;
      expect(input.value).toBe("OpenHands Cloud – Acme Inc");
    });

    await openDropdown();

    await waitFor(() => {
      expect(
        screen.getByText("OpenHands Cloud – Personal"),
      ).toBeInTheDocument();
    });
    expect(screen.getByText("OpenHands Cloud – Acme Inc")).toBeInTheDocument();
    expect(getCurrentCloudApiKey).not.toHaveBeenCalled();
  });

  it("labels an org as 'Personal Workspace' when /me reports user_id === org.id", async () => {
    const personalOrgId = "0b93b5f2-5396-49f2-8d98-61f906184270";
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        {
          id: personalOrgId,
          // The auto-generated personal-workspace org has an unfriendly
          // backend-side name; the GUI must override it.
          name: `user_${personalOrgId}_org`,
        },
        { id: "org-2", name: "Acme Inc" },
      ],
      currentOrgId: personalOrgId,
    });
    // /me for the personal org returns user_id === org_id; for the team
    // org user_id !== org_id.
    vi.mocked(getCloudOrganizationMe).mockImplementation(async (orgId) => ({
      orgId,
      userId: orgId === personalOrgId ? personalOrgId : "some-user",
      role: null,
    }));

    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend(SEED_CLOUD_PRODUCTION);
          // Land on a local backend so the cloud backend is unselected
          // and its org rows render in the dropdown.
          ctx.addBackend(SEED_LOCAL_1);
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    await openDropdown();

    await waitFor(() => {
      expect(
        screen.getByText("Production – BACKEND$PERSONAL_WORKSPACE"),
      ).toBeInTheDocument();
    });
    expect(screen.getByText("Production – Acme Inc")).toBeInTheDocument();
    // The auto-generated org name must NOT be rendered.
    expect(
      screen.queryByText(`Production – user_${personalOrgId}_org`),
    ).not.toBeInTheDocument();
  });

  it("self-heals (cloud, null) → Cloud's current org before personal workspace", async () => {
    const personalOrgId = "0b93b5f2-5396-49f2-8d98-61f906184270";
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: personalOrgId, name: "Auto-generated personal" },
        { id: "org-2", name: "Acme Inc" },
      ],
      currentOrgId: "org-2",
    });
    vi.mocked(getCloudOrganizationMe).mockResolvedValue({
      orgId: personalOrgId,
      userId: personalOrgId,
      role: null,
    });

    let cloudId = "";
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          cloudId = ctx.addBackend(SEED_CLOUD_PRODUCTION).id;
          // Simulate the post-refresh malformed state: active backend is
          // the cloud one but no orgId is set yet.
          ctx.setActive(cloudId, null);
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    // After orgs resolve, the selector snaps the active selection onto the
    // Cloud API's current_org_id locally — without round-tripping /switch on
    // the cloud backend (which would have mutated the cloud UI's current org
    // as a side effect).
    await waitFor(() => {
      const stored = JSON.parse(
        window.localStorage.getItem("openhands-active-backend") ?? "null",
      );
      expect(stored).toEqual({ backendId: cloudId, orgId: "org-2" });
    });
  });

  it("switches the active backend when an option is selected", async () => {
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend(SEED_LOCAL_1);
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    // Auto-switch lands on "Local 1"; click the seeded default to switch.
    const user = await openDropdown();
    await user.click(screen.getByText("Local"));

    const wrapper = screen.getByTestId("backend-selector");
    const input = wrapper.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("Local");
  });

  // @spec BM-002 — Switching backends keeps the user on the same page
  it.each([
    {
      name: "conversation detail → conversations list",
      startPath: "/conversations/abc",
      startRoute: "/conversations/:conversationId",
      landingRoute: "/conversations",
      expectRedirect: true,
    },
    {
      name: "automation detail → automations list",
      startPath: "/automations/abc-123",
      startRoute: "/automations/:automationId",
      landingRoute: "/automations",
      expectRedirect: true,
    },
    {
      name: "settings → stays on settings",
      startPath: "/settings",
      startRoute: "/settings",
      landingRoute: "/conversations",
      expectRedirect: false,
    },
  ])(
    "$name",
    async ({ startPath, startRoute, landingRoute, expectRedirect }) => {
      function StartRoute() {
        return (
          <TestSeed
            onMount={(ctx) => {
              ctx.addBackend(SEED_LOCAL_1);
            }}
          >
            <div data-testid="start-route" />
            <BackendSelector />
          </TestSeed>
        );
      }
      function LandingRoute() {
        return <div data-testid="landing-route" />;
      }
      const RouterStub = createRoutesStub([
        { path: startRoute, Component: StartRoute },
        { path: landingRoute, Component: LandingRoute },
      ]);
      const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      render(
        <QueryClientProvider client={queryClient}>
          <ActiveBackendProvider>
            <RouterStub initialEntries={[startPath]} />
          </ActiveBackendProvider>
        </QueryClientProvider>,
      );

      const user = await openDropdown();
      await user.click(screen.getByText("Local"));

      if (expectRedirect) {
        expect(await screen.findByTestId("landing-route")).toBeInTheDocument();
      } else {
        await waitFor(() => {
          expect(screen.getByTestId("start-route")).toBeInTheDocument();
        });
        expect(screen.queryByTestId("landing-route")).not.toBeInTheDocument();
      }
    },
  );

  it("keeps the environment-switch overlay visible even after the selector unmounts mid-switch", async () => {
    // Arrange — selector and overlay are rendered in independent trees
    // so the selector can be torn down without taking the overlay with
    // it. This mirrors production: ContextMenuContainer's click-outside
    // handler unmounts UserContextMenu (and BackendSelector) the moment
    // the dropdown's portaled option list is clicked, so the trigger
    // must not depend on BackendSelector staying mounted.
    const selectorRender = renderWithProviders(
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
        <BackendSelector />
      </TestSeed>,
    );
    render(<EnvironmentSwitchOverlay />);

    // Act — auto-switch lands on "Acme Local"; click the seeded default
    // to trigger a switch, then immediately unmount the selector (the
    // click itself would do this in production via the outside-click handler).
    const user = await openDropdown();
    await user.click(screen.getByText("Local"));
    selectorRender.unmount();

    // Assert — the overlay is still in the DOM with the chosen target
    expect(screen.getByTestId("environment-switch-overlay")).toHaveAttribute(
      "data-target",
      "Local",
    );
  });

  it("does not open backend modals on mouse down alone", async () => {
    renderWithProviders(<BackendSelector />);

    await openDropdown();

    fireEvent.mouseDown(screen.getByTestId("add-backend-menu-item"));
    expect(screen.queryByTestId("add-backend-modal")).not.toBeInTheDocument();

    fireEvent.mouseDown(screen.getByTestId("manage-backends-menu-item"));
    expect(
      screen.queryByTestId("manage-backends-modal"),
    ).not.toBeInTheDocument();
  });

  it.each([
    {
      itemTestId: "add-backend-menu-item",
      modalTestId: "add-backend-modal",
    },
    {
      itemTestId: "manage-backends-menu-item",
      modalTestId: "manage-backends-modal",
    },
  ])(
    "opens $modalTestId from touch without bubbling to surrounding menus",
    async ({ itemTestId, modalTestId }) => {
      const outsideTouchEnd = vi.fn();

      renderWithProviders(
        <div onTouchEnd={outsideTouchEnd}>
          <BackendSelector />
        </div>,
      );

      await openDropdown();
      const action = screen.getByTestId(itemTestId);

      fireEvent.touchStart(action);
      fireEvent.touchEnd(action);

      expect(outsideTouchEnd).not.toHaveBeenCalled();
      expect(await screen.findByTestId(modalTestId)).toBeInTheDocument();
    },
  );

  it("renders the backend footer actions and opens/closes the add modal", async () => {
    renderWithProviders(<BackendSelector />);

    const user = await openDropdown();
    expect(screen.getByTestId("add-backend-menu-item")).toBeInTheDocument();
    expect(screen.getByTestId("manage-backends-menu-item")).toBeInTheDocument();

    await user.click(screen.getByTestId("add-backend-menu-item"));
    expect(await screen.findByTestId("add-backend-modal")).toBeInTheDocument();

    await user.click(screen.getByTestId("add-backend-close"));
    await waitFor(() => {
      expect(screen.queryByTestId("add-backend-modal")).not.toBeInTheDocument();
    });
  });

  it("hides add-backend footer action when locked to Cloud", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");

    renderWithProviders(<BackendSelector />);

    await openDropdown();
    expect(
      screen.queryByTestId("add-backend-menu-item"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("manage-backends-menu-item")).toHaveTextContent(
      "BACKEND$RECONNECT_CLOUD",
    );
  });

  it("includes the seeded default Local backend in the manage backends modal as a removable entry", async () => {
    renderWithProviders(<BackendSelector />);

    const user = await openDropdown();
    await user.click(screen.getByTestId("manage-backends-menu-item"));

    expect(
      await screen.findByTestId("manage-backends-modal"),
    ).toBeInTheDocument();
    // The default backend is just a regular registered entry — same row
    // shape and a remove button identical to any other backend.
    expect(screen.getByTestId("manage-backends-row-Local")).toBeInTheDocument();
    expect(
      screen.getByTestId("manage-backends-remove-Local"),
    ).toBeInTheDocument();
  });

  it("opens the manage backends modal and removes a backend", async () => {
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          ctx.addBackend(SEED_LOCAL_1);
        }}
      >
        <BackendSelector />
      </TestSeed>,
    );

    const user = await openDropdown();
    await user.click(screen.getByTestId("manage-backends-menu-item"));

    expect(
      await screen.findByTestId("manage-backends-modal"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("manage-backends-row-Local 1"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("manage-backends-remove-Local 1"));
    expect(await screen.findByTestId("confirmation-modal")).toBeInTheDocument();

    await user.click(screen.getByTestId("confirm-button"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("manage-backends-row-Local 1"),
      ).not.toBeInTheDocument();
    });
    // The seeded default ("Local") survives removing only "Local 1".
    const remaining = JSON.parse(
      window.localStorage.getItem("openhands-backends") ?? "[]",
    );
    expect(remaining.map((b: { name: string }) => b.name)).toEqual(["Local"]);
  });

  // @spec BM-003 — Fallback on active backend removal
  it("shows no backend when removing the only active backend from manage backends", async () => {
    // Pre-seed the registry and active selection in localStorage so the
    // initial render already reflects `Local 1` as active. Seeding via
    // `TestSeed.onMount` instead would call `setActive` AFTER the first
    // commit, causing the Dropdown's `key={activeValue-label}` to change
    // and remount the dropdown — which races with the async health probe
    // and intermittently drops the open-click state update.
    const local1Id = "local-1-test-id";
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: local1Id,
          name: "Local 1",
          host: "http://localhost:9000",
          apiKey: "k",
          kind: "local",
        },
      ]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: local1Id, orgId: null }),
    );
    __resetActiveStoreForTests();

    renderWithProviders(<BackendSelector />);

    let wrapper = screen.getByTestId("backend-selector");
    let input = wrapper.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("Local 1");

    const user = await openDropdown();
    await user.click(screen.getByTestId("manage-backends-menu-item"));
    await user.click(
      await screen.findByTestId("manage-backends-remove-Local 1"),
    );
    await user.click(screen.getByTestId("confirm-button"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("manage-backends-row-Local 1"),
      ).not.toBeInTheDocument();
    });

    // The active selection is cleared because its target was removed;
    // no registered backend remains, so the selector shows the explicit
    // unavailable state instead of synthesizing a backend.
    const stored = JSON.parse(
      window.localStorage.getItem("openhands-active-backend") ?? "null",
    );
    expect(stored).toBeNull();

    wrapper = screen.getByTestId("backend-selector");
    input = wrapper.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("BACKEND$NO_BACKEND_AVAILABLE");
  });

  describe("connection indicator", () => {
    it("renders one status dot per option, green when the probe succeeds", async () => {
      vi.mocked(SettingsClient).mockImplementation(
        function SettingsClientMock() {
          return {
            getSettings: vi.fn().mockResolvedValue({}),
          } as unknown as SettingsClient;
        },
      );

      renderWithProviders(
        <TestSeed
          onMount={(ctx) => {
            ctx.addBackend(SEED_LOCAL_1);
          }}
        >
          <BackendSelector />
        </TestSeed>,
      );

      await openDropdown();

      // One dot for the seeded default row + one for "Local 1".
      const dots = await screen.findAllByTestId("backend-status-dot");
      expect(dots.length).toBeGreaterThanOrEqual(2);

      await waitFor(() => {
        const connected = screen
          .getAllByTestId("backend-status-dot")
          .filter((el) => el.getAttribute("data-status") === "connected");
        expect(connected.length).toBeGreaterThanOrEqual(2);
      });
    });

    it("flips the status dot to red when the local probe fails", async () => {
      vi.mocked(SettingsClient).mockImplementation(
        function SettingsClientMock() {
          return {
            getSettings: vi.fn().mockRejectedValue(new Error("ECONNREFUSED")),
          } as unknown as SettingsClient;
        },
      );

      renderWithProviders(<BackendSelector />);

      await waitFor(() => {
        const wrapper = screen.getByTestId("backend-selector");
        const dot = within(wrapper).getByTestId("backend-status-dot");
        expect(dot.getAttribute("data-status")).toBe("disconnected");
      });
    });
  });
});
