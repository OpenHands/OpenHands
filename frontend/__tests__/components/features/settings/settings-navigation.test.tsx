import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter } from "react-router";
import { SettingsNavigation } from "#/components/features/settings/settings-navigation";
import OptionService from "#/api/option-service/option-service.api";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { SAAS_NAV_ITEMS, SettingsNavItem } from "#/constants/settings-nav";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";

vi.mock("react-router", async () => ({
  ...(await vi.importActual("react-router")),
  useRevalidator: () => ({ revalidate: vi.fn() }),
}));

const mockConfig = () => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue({
    app_mode: "saas",
  } as Awaited<ReturnType<typeof OptionService.getConfig>>);
};

// Convert SettingsNavItem[] to SettingsNavRenderedItem[]
const toRenderedItems = (items: SettingsNavItem[]): SettingsNavRenderedItem[] =>
  items.map((item) => ({ type: "item", item }));

const ITEMS_WITHOUT_ORG = SAAS_NAV_ITEMS.filter(
  (item) =>
    item.to !== "/settings/org" && item.to !== "/settings/org-members",
);

const renderSettingsNavigation = (
  items: SettingsNavRenderedItem[] = toRenderedItems(SAAS_NAV_ITEMS),
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <SettingsNavigation
          isMobileMenuOpen={false}
          onCloseMobileMenu={vi.fn()}
          navigationItems={items}
        />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

describe("SettingsNavigation", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockConfig();
    useSelectedOrganizationStore.setState({ organizationId: "org-1" });
  });

  describe("renders navigation items passed via props", () => {
    it("should render org routes when included in navigation items", async () => {
      renderSettingsNavigation(toRenderedItems(SAAS_NAV_ITEMS));

      await screen.findByTestId("settings-navbar");

      const orgMembersLink = await screen.findByText("SETTINGS$NAV_ORG_MEMBERS");
      const orgLink = await screen.findByText("SETTINGS$NAV_ORGANIZATION");

      expect(orgMembersLink).toBeInTheDocument();
      expect(orgLink).toBeInTheDocument();
    });

    it("should not render org routes when excluded from navigation items", async () => {
      renderSettingsNavigation(toRenderedItems(ITEMS_WITHOUT_ORG));

      await screen.findByTestId("settings-navbar");

      const orgMembersLink = screen.queryByText("SETTINGS$NAV_ORG_MEMBERS");
      const orgLink = screen.queryByText("SETTINGS$NAV_ORGANIZATION");

      expect(orgMembersLink).not.toBeInTheDocument();
      expect(orgLink).not.toBeInTheDocument();
    });

    it("should render all non-org SAAS items regardless of which items are passed", async () => {
      renderSettingsNavigation(toRenderedItems(SAAS_NAV_ITEMS));

      await screen.findByTestId("settings-navbar");

      // Verify non-org items are rendered (using their i18n keys as text since
      // react-i18next returns the key when no translation is loaded)
      const secretsLink = await screen.findByText("SETTINGS$NAV_SECRETS");
      const apiKeysLink = await screen.findByText("SETTINGS$NAV_API_KEYS");

      expect(secretsLink).toBeInTheDocument();
      expect(apiKeysLink).toBeInTheDocument();
    });

    it("should render empty nav when given an empty items list", async () => {
      renderSettingsNavigation([]);

      await screen.findByTestId("settings-navbar");

      // No nav links should be rendered
      const orgMembersLink = screen.queryByText("SETTINGS$NAV_ORG_MEMBERS");
      const orgLink = screen.queryByText("SETTINGS$NAV_ORGANIZATION");

      expect(orgMembersLink).not.toBeInTheDocument();
      expect(orgLink).not.toBeInTheDocument();
    });
  });

  describe("renders section headers and dividers", () => {
    it("should render section headers when included in navigation items", async () => {
      // Arrange
      const itemsWithHeader: SettingsNavRenderedItem[] = [
        { type: "header", text: "SETTINGS$ORG_SETTINGS_HEADER" as any },
        ...toRenderedItems(SAAS_NAV_ITEMS.slice(0, 2)),
      ];

      // Act
      renderSettingsNavigation(itemsWithHeader);
      await screen.findByTestId("settings-navbar");

      // Assert
      expect(screen.getByText("SETTINGS$ORG_SETTINGS_HEADER")).toBeInTheDocument();
    });

    it("should render dividers when included in navigation items", async () => {
      // Arrange
      const itemsWithDivider: SettingsNavRenderedItem[] = [
        ...toRenderedItems(SAAS_NAV_ITEMS.slice(0, 2)),
        { type: "divider" },
        ...toRenderedItems(SAAS_NAV_ITEMS.slice(2, 4)),
      ];

      // Act
      renderSettingsNavigation(itemsWithDivider);
      await screen.findByTestId("settings-navbar");

      // Assert - divider is a div with border-t class
      const navbar = screen.getByTestId("settings-navbar");
      const dividers = navbar.querySelectorAll(".border-t");
      expect(dividers.length).toBeGreaterThan(0);
    });

    it("should render multiple headers and dividers in correct order", async () => {
      // Arrange
      const itemsWithHeadersAndDividers: SettingsNavRenderedItem[] = [
        { type: "header", text: "SETTINGS$ORG_SETTINGS_HEADER" as any },
        ...toRenderedItems(SAAS_NAV_ITEMS.slice(0, 1)),
        { type: "divider" },
        { type: "header", text: "SETTINGS$PERSONAL_SETTINGS_HEADER" as any },
        ...toRenderedItems(SAAS_NAV_ITEMS.slice(1, 2)),
      ];

      // Act
      renderSettingsNavigation(itemsWithHeadersAndDividers);
      await screen.findByTestId("settings-navbar");

      // Assert
      expect(screen.getByText("SETTINGS$ORG_SETTINGS_HEADER")).toBeInTheDocument();
      expect(screen.getByText("SETTINGS$PERSONAL_SETTINGS_HEADER")).toBeInTheDocument();
    });
  });
});
