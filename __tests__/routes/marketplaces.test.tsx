import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import MarketplacesScreen from "#/routes/marketplaces";
import SettingsService from "#/api/settings-service/settings-service.api";
import MarketplacesService from "#/api/marketplaces-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type {
  CatalogEntry,
  MarketplaceCatalog,
  MarketplaceSource,
} from "#/types/marketplaces";

const navigateMock = vi.fn();
let routeMarketplaceName: string | undefined;

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    navigate: navigateMock,
    currentPath: "/marketplaces",
    conversationId: null,
    isNavigating: false,
  }),
  NavigationProvider: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("react-router", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router")>();
  return {
    ...actual,
    useParams: () => ({ marketplaceName: routeMarketplaceName }),
  };
});

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8001",
  apiKey: "",
  kind: "local",
};

function marketplace(
  overrides: Partial<MarketplaceSource> = {},
): MarketplaceSource {
  return {
    name: "openhands-extensions",
    source: "github:OpenHands/extensions",
    ref: "main",
    repo_path: null,
    scope: "instance",
    trusted: true,
    token_secret: null,
    is_default: true,
    resolved_ref: "fb020c71",
    last_refreshed_at: "2026-11-06T09:00:00Z",
    counts: { mcp: 1, skill: 1, plugin: 1, app: 0, automation: 1 },
    error_count: 0,
    ...overrides,
  };
}

const DEFAULT = marketplace();
const ACME = marketplace({
  name: "acme-internal",
  source: "github:acme-corp/extensions",
  scope: "user",
  trusted: false,
  is_default: false,
  counts: { mcp: 1, skill: 1, plugin: 1, app: 0, automation: 1 },
  error_count: 1,
});

function entry(
  source: MarketplaceSource,
  overrides: Partial<CatalogEntry>,
): CatalogEntry {
  const kind = overrides.kind ?? "skill";
  const name = overrides.name ?? "entry";
  return {
    id: `${kind}:${name}@${source.name}`,
    kind,
    name,
    marketplace: source.name,
    description: `${name} description`,
    tags: ["ops"],
    version: null,
    runs_code: false,
    also_in: [],
    install: {
      state: "not_installed",
      enabled: false,
      installed_ref: null,
      policy: "track",
    },
    requires: [],
    transport: null,
    icon: null,
    ...overrides,
  };
}

function catalogFor(source: MarketplaceSource): MarketplaceCatalog {
  const prefix = source.name === ACME.name ? "acme-" : "";
  return {
    marketplace: source,
    entries: [
      entry(source, { kind: "skill", name: `${prefix}review` }),
      entry(source, {
        kind: "plugin",
        name: `${prefix}deploy`,
        runs_code: true,
      }),
      entry(source, {
        kind: "mcp",
        name: `${prefix}ledger`,
        transport: "http",
      }),
      entry(source, {
        kind: "automation",
        name: `${prefix}sweep`,
        requires: ["github"],
        install: {
          state: "update_available",
          enabled: true,
          installed_ref: "1a2b3c4d",
          policy: "track",
        },
      }),
    ],
    errors:
      source.name === ACME.name
        ? [
            {
              entry: "nightly-cve",
              kind: "automation",
              message: "Invalid manifest",
            },
          ]
        : [],
  };
}

function renderScreen() {
  return render(<MarketplacesScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

describe("MarketplacesScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    navigateMock.mockReset();
    routeMarketplaceName = undefined;
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
    vi.spyOn(MarketplacesService, "list").mockResolvedValue([DEFAULT, ACME]);
    vi.spyOn(MarketplacesService, "getCatalog").mockImplementation(
      async (name) => catalogFor(name === ACME.name ? ACME : DEFAULT),
    );
  });

  it("selects the default marketplace and shows every kind together, without the Customize rail", async () => {
    renderScreen();

    expect(
      await screen.findByTestId("catalog-entry-skill-review"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("catalog-entry-plugin-deploy"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("catalog-entry-mcp-ledger")).toBeInTheDocument();
    expect(
      screen.getByTestId("catalog-entry-automation-sweep"),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("MARKETPLACES$PICKER_LABEL")).toHaveValue(
      "openhands-extensions · MARKETPLACES$SCOPE_INSTANCE",
    );
    expect(
      screen.queryByTestId("extensions-navbar-desktop"),
    ).not.toBeInTheDocument();
    // The default cannot be removed.
    expect(screen.queryByTestId("marketplace-remove")).not.toBeInTheDocument();
    expect(
      screen.getByTestId("marketplace-tab-app").closest("button"),
    ).toBeDisabled();
  });

  it("switches kinds with the tabs and narrows with the facet menu", async () => {
    const user = userEvent.setup();
    renderScreen();
    await screen.findByTestId("catalog-entry-skill-review");

    await user.click(screen.getByTestId("marketplace-tab-plugin"));
    expect(
      screen.getByTestId("catalog-entry-plugin-deploy"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("catalog-entry-skill-review"),
    ).not.toBeInTheDocument();
    expect(screen.getAllByTestId(/^catalog-entry-kind-/)).toHaveLength(1);

    await user.click(screen.getByTestId("marketplace-tab-all"));
    await user.click(screen.getByTestId("marketplace-facet-status-installed"));
    expect(
      screen.getByTestId("catalog-entry-automation-sweep"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("catalog-entry-skill-review"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("marketplace-clear-filters"));
    await user.type(screen.getByTestId("marketplace-search-input"), "ledger");
    expect(screen.getByTestId("catalog-entry-mcp-ledger")).toBeInTheDocument();
    expect(
      screen.queryByTestId("catalog-entry-plugin-deploy"),
    ).not.toBeInTheDocument();
  });

  it("shows the selected marketplace from the URL with its errors, trust gate and Remove", async () => {
    const user = userEvent.setup();
    routeMarketplaceName = ACME.name;
    const update = vi
      .spyOn(MarketplacesService, "update")
      .mockResolvedValue({ ...ACME, trusted: true });

    renderScreen();

    expect(
      await screen.findByTestId("catalog-entry-plugin-acme-deploy"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("marketplace-catalog-errors")).toHaveTextContent(
      "nightly-cve",
    );
    expect(screen.getByTestId("marketplace-remove")).toBeInTheDocument();
    expect(
      screen.getByTestId("catalog-entry-install-plugin-acme-deploy"),
    ).toBeDisabled();
    expect(
      screen.getByTestId("catalog-entry-install-skill-acme-review"),
    ).toBeEnabled();

    await user.click(
      within(screen.getByTestId("marketplace-trust-control")).getByRole(
        "switch",
      ),
    );
    await waitFor(() =>
      expect(update).toHaveBeenCalledWith(ACME.name, { trusted: true }),
    );
  });

  it("installs an entry with its marketplace and id", async () => {
    const user = userEvent.setup();
    const install = vi
      .spyOn(MarketplacesService, "installEntry")
      .mockResolvedValue(entry(DEFAULT, { name: "review" }));

    renderScreen();
    await user.click(
      await screen.findByTestId("catalog-entry-install-skill-review"),
    );

    await waitFor(() =>
      expect(install).toHaveBeenCalledWith(
        DEFAULT.name,
        "skill:review@openhands-extensions",
      ),
    );
  });

  it("adds a marketplace from the header and selects it", async () => {
    const user = userEvent.setup();
    const add = vi.spyOn(MarketplacesService, "add").mockResolvedValue(
      marketplace({
        name: "superpowers",
        source: "github:obra/superpowers",
        scope: "user",
        is_default: false,
      }),
    );

    renderScreen();
    await user.click(await screen.findByTestId("marketplaces-add-button"));
    await user.type(
      screen.getByTestId("add-marketplace-source-input"),
      "github:obra/superpowers",
    );
    await user.click(screen.getByTestId("add-marketplace-submit"));

    await waitFor(() =>
      expect(add).toHaveBeenCalledWith({
        source: "github:obra/superpowers",
        name: undefined,
        ref: null,
        trusted: false,
        token_secret: null,
      }),
    );
    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/marketplaces/superpowers"),
    );
  });

  it("removes the selected marketplace after confirmation and returns to the default", async () => {
    const user = userEvent.setup();
    routeMarketplaceName = ACME.name;
    const remove = vi
      .spyOn(MarketplacesService, "remove")
      .mockResolvedValue(undefined);

    renderScreen();
    await user.click(await screen.findByTestId("marketplace-remove"));
    await user.click(screen.getByTestId("confirm-button"));

    await waitFor(() => expect(remove).toHaveBeenCalledWith(ACME.name));
    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/marketplaces"),
    );
  });
});
