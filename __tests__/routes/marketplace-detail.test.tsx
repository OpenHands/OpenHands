import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import MarketplaceDetailScreen from "#/routes/marketplace-detail";
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

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    navigate: navigateMock,
    currentPath: "/marketplaces/acme-internal",
    conversationId: null,
    isNavigating: false,
  }),
  NavigationProvider: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("react-router", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router")>();
  return { ...actual, useParams: () => ({ marketplaceName: "acme-internal" }) };
});

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8001",
  apiKey: "",
  kind: "local",
};

const marketplace: MarketplaceSource = {
  name: "acme-internal",
  source: "github:acme-corp/extensions",
  ref: "main",
  repo_path: null,
  scope: "user",
  trusted: false,
  token_secret: null,
  is_default: false,
  resolved_ref: "4e1c9b2d",
  last_refreshed_at: "2026-11-06T09:00:00Z",
  counts: { mcp: 1, skill: 1, plugin: 1, app: 0, automation: 1 },
  error_count: 1,
};

function entry(overrides: Partial<CatalogEntry>): CatalogEntry {
  const base: CatalogEntry = {
    id: "skill:acme-oncall@acme-internal",
    kind: "skill",
    name: "acme-oncall",
    marketplace: "acme-internal",
    description: "On-call runbook",
    tags: ["operations"],
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
  };
  return {
    ...base,
    ...overrides,
    id: `${overrides.kind ?? base.kind}:${overrides.name ?? base.name}@acme-internal`,
  };
}

const catalog: MarketplaceCatalog = {
  marketplace,
  entries: [
    entry({}),
    entry({ kind: "plugin", name: "acme-deploy", runs_code: true }),
    entry({ kind: "mcp", name: "acme-ledger", transport: "http" }),
    entry({
      kind: "automation",
      name: "acme-weekly-cve-sweep",
      requires: ["github", "slack"],
      install: {
        state: "update_available",
        enabled: true,
        installed_ref: "1a2b3c4d",
        policy: "track",
      },
    }),
  ],
  errors: [
    { entry: "nightly-cve", kind: "automation", message: "Invalid manifest" },
  ],
};

function renderScreen() {
  return render(<MarketplaceDetailScreen />, {
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

describe("MarketplaceDetailScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    navigateMock.mockReset();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
    vi.spyOn(MarketplacesService, "getCatalog").mockResolvedValue(catalog);
  });

  it("shows every kind together, the read errors, and per-kind actions", async () => {
    renderScreen();

    expect(
      await screen.findByTestId("catalog-entry-skill-acme-oncall"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("catalog-entry-plugin-acme-deploy"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("catalog-entry-mcp-acme-ledger"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("catalog-entry-automation-acme-weekly-cve-sweep"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("marketplace-catalog-errors")).toHaveTextContent(
      "nightly-cve",
    );

    // Kind decides what install means.
    expect(
      screen.getByTestId("catalog-entry-install-mcp-acme-ledger"),
    ).toHaveTextContent("MARKETPLACES$ACTION_ADD_MCP");
    expect(
      screen.getByTestId("catalog-entry-install-skill-acme-oncall"),
    ).toHaveTextContent("MARKETPLACES$ACTION_INSTALL");
    expect(
      screen.getByTestId(
        "catalog-entry-update-automation-acme-weekly-cve-sweep",
      ),
    ).toBeInTheDocument();
  });

  it("blocks installing content that runs code until the marketplace is trusted", async () => {
    const user = userEvent.setup();
    const update = vi
      .spyOn(MarketplacesService, "update")
      .mockResolvedValue({ ...marketplace, trusted: true });

    renderScreen();

    const install = await screen.findByTestId(
      "catalog-entry-install-plugin-acme-deploy",
    );
    expect(install).toBeDisabled();
    expect(
      screen.getByTestId("catalog-entry-install-skill-acme-oncall"),
    ).toBeEnabled();

    await user.click(screen.getByRole("switch"));
    await waitFor(() =>
      expect(update).toHaveBeenCalledWith("acme-internal", { trusted: true }),
    );
  });

  it("filters the catalog by kind and by search", async () => {
    const user = userEvent.setup();
    renderScreen();
    await screen.findByTestId("catalog-entry-skill-acme-oncall");

    await user.click(screen.getByTestId("marketplace-kind-plugin"));
    expect(
      screen.getByTestId("catalog-entry-plugin-acme-deploy"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("catalog-entry-skill-acme-oncall"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("marketplace-kind-app")).toBeDisabled();

    await user.click(screen.getByTestId("marketplace-kind-all"));
    await user.type(screen.getByTestId("marketplace-search-input"), "ledger");
    expect(
      screen.getByTestId("catalog-entry-mcp-acme-ledger"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("catalog-entry-plugin-acme-deploy"),
    ).not.toBeInTheDocument();
  });

  it("installs an entry through the service with its marketplace and id", async () => {
    const user = userEvent.setup();
    const install = vi
      .spyOn(MarketplacesService, "installEntry")
      .mockResolvedValue(entry({}));

    renderScreen();
    await user.click(
      await screen.findByTestId("catalog-entry-install-skill-acme-oncall"),
    );

    await waitFor(() =>
      expect(install).toHaveBeenCalledWith(
        "acme-internal",
        "skill:acme-oncall@acme-internal",
      ),
    );
  });
});
