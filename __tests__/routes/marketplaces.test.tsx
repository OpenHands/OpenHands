import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
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
import type { MarketplaceSource } from "#/types/marketplaces";

const navigateMock = vi.fn();

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    navigate: navigateMock,
    currentPath: "/marketplaces",
    conversationId: null,
    isNavigating: false,
  }),
  NavigationProvider: ({ children }: { children: React.ReactNode }) => children,
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8001",
  apiKey: "",
  kind: "local",
};

function buildMarketplace(
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
    counts: { mcp: 79, skill: 70, plugin: 9, app: 3, automation: 12 },
    error_count: 0,
    ...overrides,
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
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
  });

  it("groups marketplaces by scope and shows each kind's count", async () => {
    vi.spyOn(MarketplacesService, "list").mockResolvedValue([
      buildMarketplace(),
      buildMarketplace({
        name: "acme-internal",
        source: "github:acme-corp/extensions",
        scope: "user",
        trusted: false,
        is_default: false,
        counts: { mcp: 2, skill: 2, plugin: 1, app: 1, automation: 1 },
        error_count: 1,
      }),
    ]);

    renderScreen();

    expect(
      await screen.findByTestId("marketplace-card-openhands-extensions"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("marketplaces-scope-instance")).toContainElement(
      screen.getByTestId("marketplace-card-openhands-extensions"),
    );
    expect(screen.getByTestId("marketplaces-scope-user")).toContainElement(
      screen.getByTestId("marketplace-card-acme-internal"),
    );
    expect(
      screen.getByTestId("marketplace-count-openhands-extensions-mcp"),
    ).toHaveTextContent("79");
    expect(
      screen.getByTestId("marketplace-trust-acme-internal"),
    ).toHaveTextContent("MARKETPLACES$UNTRUSTED_BADGE");
    // The default marketplace cannot be removed; a user one can.
    expect(
      screen.queryByTestId("marketplace-remove-openhands-extensions"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("marketplace-remove-acme-internal"),
    ).toBeInTheDocument();
  });

  it("opens a marketplace's own page when its card is clicked", async () => {
    const user = userEvent.setup();
    vi.spyOn(MarketplacesService, "list").mockResolvedValue([
      buildMarketplace({ name: "acme internal", is_default: false }),
    ]);

    renderScreen();
    await user.click(
      await screen.findByTestId("marketplace-card-acme internal"),
    );

    expect(navigateMock).toHaveBeenCalledWith("/marketplaces/acme%20internal");
  });

  it("adds a marketplace from the form and navigates to it", async () => {
    const user = userEvent.setup();
    vi.spyOn(MarketplacesService, "list").mockResolvedValue([
      buildMarketplace(),
    ]);
    const add = vi.spyOn(MarketplacesService, "add").mockResolvedValue(
      buildMarketplace({
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

    await waitFor(() => expect(add).toHaveBeenCalledTimes(1));
    expect(add).toHaveBeenCalledWith({
      source: "github:obra/superpowers",
      name: undefined,
      ref: null,
      trusted: false,
      token_secret: null,
    });
    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/marketplaces/superpowers"),
    );
  });

  it("removes a user marketplace after confirmation", async () => {
    const user = userEvent.setup();
    vi.spyOn(MarketplacesService, "list").mockResolvedValue([
      buildMarketplace({
        name: "acme-internal",
        scope: "user",
        is_default: false,
      }),
    ]);
    const remove = vi
      .spyOn(MarketplacesService, "remove")
      .mockResolvedValue(undefined);

    renderScreen();
    await user.click(
      await screen.findByTestId("marketplace-remove-acme-internal"),
    );
    await user.click(screen.getByTestId("confirm-button"));

    await waitFor(() => expect(remove).toHaveBeenCalledWith("acme-internal"));
  });
});
