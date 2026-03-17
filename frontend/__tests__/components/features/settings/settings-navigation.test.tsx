import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter } from "react-router";
import { SettingsNavigation } from "#/components/features/settings/settings-navigation";
import OptionService from "#/api/option-service/option-service.api";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { SAAS_NAV_ITEMS, SettingsNavItem } from "#/constants/settings-nav";

vi.mock("react-router", async () => ({
  ...(await vi.importActual("react-router")),
  useRevalidator: () => ({ revalidate: vi.fn() }),
}));

const mockConfig = () => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue({
    app_mode: "saas",
  } as Awaited<ReturnType<typeof OptionService.getConfig>>);
};

const ITEMS_WITHOUT_ORG = SAAS_NAV_ITEMS.filter(
  (item) =>
    item.to !== "/settings/org" && item.to !== "/settings/org-members",
);

const renderSettingsNavigation = (
  items: SettingsNavItem[] = SAAS_NAV_ITEMS,
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
      renderSettingsNavigation(SAAS_NAV_ITEMS);

      await screen.findByTestId("settings-navbar");

      const orgMembersLink = await screen.findByText("Organization Members");
      const orgLink = await screen.findByText("Organization");

      expect(orgMembersLink).toBeInTheDocument();
      expect(orgLink).toBeInTheDocument();
    });

    it("should not render org routes when excluded from navigation items", async () => {
      renderSettingsNavigation(ITEMS_WITHOUT_ORG);

      await screen.findByTestId("settings-navbar");

      const orgMembersLink = screen.queryByText("Organization Members");
      const orgLink = screen.queryByText("Organization");

      expect(orgMembersLink).not.toBeInTheDocument();
      expect(orgLink).not.toBeInTheDocument();
    });

    it("should render all non-org SAAS items regardless of which items are passed", async () => {
      renderSettingsNavigation(SAAS_NAV_ITEMS);

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
      const orgMembersLink = screen.queryByText("Organization Members");
      const orgLink = screen.queryByText("Organization");

      expect(orgMembersLink).not.toBeInTheDocument();
      expect(orgLink).not.toBeInTheDocument();
    });
  });
});
