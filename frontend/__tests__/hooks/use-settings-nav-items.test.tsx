import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import OptionService from "#/api/option-service/option-service.api";
import {
  useSettingsNavItems,
  SettingsNavRenderedItem,
} from "#/hooks/use-settings-nav-items";
import { WebClientFeatureFlags } from "#/api/option-service/option.types";

// Helper to find an item by path in rendered items
const findItemByPath = (
  items: SettingsNavRenderedItem[],
  path: string,
): SettingsNavRenderedItem | undefined =>
  items.find((item) => item.type === "item" && item.item.to === path);

// Helper to get only nav items (excluding headers and dividers)
const getNavItems = (items: SettingsNavRenderedItem[]) =>
  items.filter((item) => item.type === "item");

// Mock useOrgTypeAndAccess
const mockOrgTypeAndAccess = vi.hoisted(() => ({
  isPersonalOrg: false,
  isTeamOrg: false,
  organizationId: null as string | null,
  selectedOrg: null,
  canViewOrgRoutes: false,
}));

vi.mock("#/hooks/use-org-type-and-access", () => ({
  useOrgTypeAndAccess: () => mockOrgTypeAndAccess,
}));

// Mock useMe
const mockMe = vi.hoisted(() => ({
  data: null as { role: string } | null | undefined,
}));

vi.mock("#/hooks/query/use-me", () => ({
  useMe: () => mockMe,
}));

const queryClient = new QueryClient();
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

const mockConfig = (
  appMode: "saas" | "oss",
  hideLlmSettings = false,
  enableBilling = true,
) => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue({
    app_mode: appMode,
    feature_flags: {
      hide_llm_settings: hideLlmSettings,
      enable_billing: enableBilling,
      enable_jira: false,
      enable_jira_dc: false,
      enable_linear: false,
    },
  } as Awaited<ReturnType<typeof OptionService.getConfig>>);
};

vi.mock("react-router", () => ({
  useRevalidator: () => ({ revalidate: vi.fn() }),
}));

const mockConfigWithFeatureFlags = (
  appMode: "saas" | "oss",
  featureFlags: Partial<WebClientFeatureFlags>,
) => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue({
    app_mode: appMode,
    feature_flags: {
      enable_billing: true, // Enable billing by default so it's not hidden
      hide_llm_settings: false,
      enable_jira: false,
      enable_jira_dc: false,
      enable_linear: false,
      hide_users_page: false,
      hide_billing_page: false,
      hide_integrations_page: false,
      ...featureFlags,
    },
  } as Awaited<ReturnType<typeof OptionService.getConfig>>);
};

describe("useSettingsNavItems", () => {
  beforeEach(() => {
    queryClient.clear();
    vi.restoreAllMocks();
    // Reset mock state
    mockOrgTypeAndAccess.isPersonalOrg = false;
    mockOrgTypeAndAccess.isTeamOrg = false;
    mockOrgTypeAndAccess.organizationId = null;
    mockOrgTypeAndAccess.selectedOrg = null;
    mockOrgTypeAndAccess.canViewOrgRoutes = false;
    mockMe.data = null;
  });

  it("should return SAAS_NAV_ITEMS minus billing/org/org-members when userRole is 'member'", async () => {
    mockConfig("saas");
    mockMe.data = { role: "member" };
    mockOrgTypeAndAccess.organizationId = "org-1";

    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      // Members should not see billing, org, or org-members routes
      expect(findItemByPath(result.current, "/settings/billing")).toBeUndefined();
      expect(findItemByPath(result.current, "/settings/org")).toBeUndefined();
      expect(findItemByPath(result.current, "/settings/org-members")).toBeUndefined();
      // Personal LLM/Condenser/Verification routes are hidden in SaaS;
      // members see the org-defaults equivalents (read-only on the page itself).
      expect(findItemByPath(result.current, "/settings")).toBeUndefined();
      expect(
        findItemByPath(result.current, "/settings/org-defaults"),
      ).toBeDefined();
      expect(findItemByPath(result.current, "/settings/user")).toBeDefined();
    });
  });

  it("should return OSS_NAV_ITEMS when app_mode is 'oss'", async () => {
    mockConfig("oss");
    mockMe.data = { role: "admin" };
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      // OSS mode should return items matching OSS_NAV_ITEMS paths
      const navItems = getNavItems(result.current);
      const ossPaths = OSS_NAV_ITEMS.map((item) => item.to);
      const resultPaths = navItems.map((item) =>
        item.type === "item" ? item.item.to : null,
      );
      expect(resultPaths).toEqual(ossPaths);
    });
  });

  it("should filter out '/settings' item when hide_llm_settings feature flag is enabled", async () => {
    mockConfig("saas", true);
    mockMe.data = { role: "admin" };
    mockOrgTypeAndAccess.organizationId = "org-1";
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(findItemByPath(result.current, "/settings")).toBeUndefined();
    });
  });

  describe("org-type and role-based filtering", () => {
    it("should include org routes by default for team org admin", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isTeamOrg = true;
      mockOrgTypeAndAccess.organizationId = "org-123";
      mockMe.data = { role: "admin" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      // Wait for config to load (check that any SAAS item is present)
      await waitFor(() => {
        expect(result.current.length).toBeGreaterThan(0);
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be included for team org admin
      expect(
        findItemByPath(result.current, "/settings/org"),
      ).toBeDefined();
      expect(
        findItemByPath(result.current, "/settings/org-members"),
      ).toBeDefined();
    });

    it("should hide org routes when isPersonalOrg is true", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isPersonalOrg = true;
      mockOrgTypeAndAccess.organizationId = "org-123";
      mockMe.data = { role: "admin" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      // Wait for config to load (check that any SAAS item is present)
      await waitFor(() => {
        expect(result.current.length).toBeGreaterThan(0);
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be filtered out for personal orgs
      expect(
        findItemByPath(result.current, "/settings/org"),
      ).toBeUndefined();
      expect(
        findItemByPath(result.current, "/settings/org-members"),
      ).toBeUndefined();
    });

    it("should hide org routes when user role is member", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isTeamOrg = true;
      mockOrgTypeAndAccess.organizationId = "org-123";
      mockMe.data = { role: "member" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      // Wait for config to load
      await waitFor(() => {
        expect(result.current.length).toBeGreaterThan(0);
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be hidden for members
      expect(
        findItemByPath(result.current, "/settings/org"),
      ).toBeUndefined();
      expect(
        findItemByPath(result.current, "/settings/org-members"),
      ).toBeUndefined();
    });

    it("should hide org routes when no organization is selected", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isTeamOrg = false;
      mockOrgTypeAndAccess.isPersonalOrg = false;
      mockOrgTypeAndAccess.organizationId = null;
      mockMe.data = { role: "admin" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      // Wait for config to load
      await waitFor(() => {
        expect(result.current.length).toBeGreaterThan(0);
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be hidden when no org is selected
      expect(
        findItemByPath(result.current, "/settings/org"),
      ).toBeUndefined();
      expect(
        findItemByPath(result.current, "/settings/org-members"),
      ).toBeUndefined();
    });

    it("should hide billing route when isTeamOrg is true", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isTeamOrg = true;
      mockOrgTypeAndAccess.organizationId = "org-123";
      mockMe.data = { role: "admin" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      // Wait for config to load
      await waitFor(() => {
        expect(result.current.length).toBeGreaterThan(0);
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
      });

      // Billing should be hidden for team orgs
      expect(
        findItemByPath(result.current, "/settings/billing"),
      ).toBeUndefined();
    });

    it("should show billing route for personal org", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isPersonalOrg = true;
      mockOrgTypeAndAccess.isTeamOrg = false;
      mockOrgTypeAndAccess.organizationId = "org-123";
      mockMe.data = { role: "admin" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      // Wait for config to load
      await waitFor(() => {
        expect(result.current.length).toBeGreaterThan(0);
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
      });

      // Billing should be visible for personal orgs
      expect(
        findItemByPath(result.current, "/settings/billing"),
      ).toBeDefined();
    });
  });

  describe("hide page feature flags", () => {
    beforeEach(() => {
      // Set up user as admin with org context so billing is accessible
      mockMe.data = { role: "admin" };
      mockOrgTypeAndAccess.isPersonalOrg = true; // Personal org shows billing
      mockOrgTypeAndAccess.isTeamOrg = false;
      mockOrgTypeAndAccess.organizationId = "org-1";
    });

    it("should filter out '/settings/user' when hide_users_page is true", async () => {
      mockConfigWithFeatureFlags("saas", { hide_users_page: true });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeUndefined();
        // Other pages should still be present
        expect(
          findItemByPath(result.current, "/settings/integrations"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/billing"),
        ).toBeDefined();
      });
    });

    it("should filter out '/settings/billing' when hide_billing_page is true", async () => {
      mockConfigWithFeatureFlags("saas", { hide_billing_page: true });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings/billing"),
        ).toBeUndefined();
        // Other pages should still be present
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/integrations"),
        ).toBeDefined();
      });
    });

    it("should filter out '/settings/integrations' when hide_integrations_page is true", async () => {
      mockConfigWithFeatureFlags("saas", { hide_integrations_page: true });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings/integrations"),
        ).toBeUndefined();
        // Other pages should still be present
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/billing"),
        ).toBeDefined();
      });
    });

    it("should filter out multiple pages when multiple flags are true", async () => {
      mockConfigWithFeatureFlags("saas", {
        hide_users_page: true,
        hide_billing_page: true,
        hide_integrations_page: true,
      });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/billing"),
        ).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/integrations"),
        ).toBeUndefined();
        // Personal LLM is hidden in SaaS; the org-defaults equivalent
        // shows up instead (an org is selected in this test's setup).
        expect(
          findItemByPath(result.current, "/settings"),
        ).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/org-defaults"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/app"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/secrets"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/mcp"),
        ).toBeDefined();
      });
    });

    it("should keep all pages visible when no hide flags are set", async () => {
      mockConfigWithFeatureFlags("saas", {});
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        // All SAAS pages should be present
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/billing"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/integrations"),
        ).toBeDefined();
        // Personal LLM is hidden in SaaS; users see /settings/org-defaults instead.
        expect(
          findItemByPath(result.current, "/settings"),
        ).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/org-defaults"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/app"),
        ).toBeDefined();
      });
    });

    it("should filter out '/settings/integrations' in OSS mode when hide_integrations_page is true", async () => {
      mockConfigWithFeatureFlags("oss", { hide_integrations_page: true });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings/integrations"),
        ).toBeUndefined();
        // Other OSS pages should still be present
        expect(
          findItemByPath(result.current, "/settings"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/mcp"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/app"),
        ).toBeDefined();
      });
    });

    it("hides personal LLM/Condenser/Verification when org-defaults versions are visible (team-org admin)", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isTeamOrg = true;
      mockOrgTypeAndAccess.isPersonalOrg = false;
      mockOrgTypeAndAccess.organizationId = "org-123";
      mockMe.data = { role: "admin" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings/org-defaults"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/org-defaults/condenser"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/org-defaults/verification"),
        ).toBeDefined();

        expect(findItemByPath(result.current, "/settings")).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/condenser"),
        ).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/verification"),
        ).toBeUndefined();
      });
    });

    it("hides personal LLM/Condenser/Verification for any user in a non-personal org (team-org member)", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isTeamOrg = true;
      mockOrgTypeAndAccess.isPersonalOrg = false;
      mockOrgTypeAndAccess.organizationId = "org-123";
      mockMe.data = { role: "member" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings/user"),
        ).toBeDefined();
        expect(findItemByPath(result.current, "/settings")).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/condenser"),
        ).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/verification"),
        ).toBeUndefined();
      });
    });

    it("keeps personal LLM/Condenser/Verification visible in the personal workspace", async () => {
      mockConfig("saas");
      mockOrgTypeAndAccess.isTeamOrg = false;
      mockOrgTypeAndAccess.isPersonalOrg = true;
      mockOrgTypeAndAccess.organizationId = "org-personal";
      mockMe.data = { role: "admin" };

      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(findItemByPath(result.current, "/settings")).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/condenser"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/verification"),
        ).toBeDefined();
      });
    });

    it("should filter out both LLM and integrations when both flags are true in OSS mode", async () => {
      mockConfigWithFeatureFlags("oss", {
        hide_llm_settings: true,
        hide_integrations_page: true,
      });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          findItemByPath(result.current, "/settings"),
        ).toBeUndefined();
        expect(
          findItemByPath(result.current, "/settings/integrations"),
        ).toBeUndefined();
        // Other OSS pages should still be present
        expect(
          findItemByPath(result.current, "/settings/mcp"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/app"),
        ).toBeDefined();
        expect(
          findItemByPath(result.current, "/settings/secrets"),
        ).toBeDefined();
      });
    });
  });
});
