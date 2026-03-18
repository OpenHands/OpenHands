import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import OptionService from "#/api/option-service/option-service.api";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { WebClientFeatureFlags } from "#/api/option-service/option.types";

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
      expect(result.current).toEqual(
        SAAS_NAV_ITEMS.filter(
          (item) =>
            item.to !== "/settings/billing" &&
            item.to !== "/settings/org" &&
            item.to !== "/settings/org-members",
        ),
      );
    });
  });

  it("should return OSS_NAV_ITEMS when app_mode is 'oss'", async () => {
    mockConfig("oss");
    mockMe.data = { role: "admin" };
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(result.current).toEqual(OSS_NAV_ITEMS);
    });
  });

  it("should filter out '/settings' item when hide_llm_settings feature flag is enabled", async () => {
    mockConfig("saas", true);
    mockMe.data = { role: "admin" };
    mockOrgTypeAndAccess.organizationId = "org-1";
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(
        result.current.find((item) => item.to === "/settings"),
      ).toBeUndefined();
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be included for team org admin
      expect(
        result.current.find((item) => item.to === "/settings/org"),
      ).toBeDefined();
      expect(
        result.current.find((item) => item.to === "/settings/org-members"),
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be filtered out for personal orgs
      expect(
        result.current.find((item) => item.to === "/settings/org"),
      ).toBeUndefined();
      expect(
        result.current.find((item) => item.to === "/settings/org-members"),
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be hidden for members
      expect(
        result.current.find((item) => item.to === "/settings/org"),
      ).toBeUndefined();
      expect(
        result.current.find((item) => item.to === "/settings/org-members"),
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
      });

      // Org routes should be hidden when no org is selected
      expect(
        result.current.find((item) => item.to === "/settings/org"),
      ).toBeUndefined();
      expect(
        result.current.find((item) => item.to === "/settings/org-members"),
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
      });

      // Billing should be hidden for team orgs
      expect(
        result.current.find((item) => item.to === "/settings/billing"),
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
      });

      // Billing should be visible for personal orgs
      expect(
        result.current.find((item) => item.to === "/settings/billing"),
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeUndefined();
        // Other pages should still be present
        expect(
          result.current.find((item) => item.to === "/settings/integrations"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/billing"),
        ).toBeDefined();
      });
    });

    it("should filter out '/settings/billing' when hide_billing_page is true", async () => {
      mockConfigWithFeatureFlags("saas", { hide_billing_page: true });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          result.current.find((item) => item.to === "/settings/billing"),
        ).toBeUndefined();
        // Other pages should still be present
        expect(
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/integrations"),
        ).toBeDefined();
      });
    });

    it("should filter out '/settings/integrations' when hide_integrations_page is true", async () => {
      mockConfigWithFeatureFlags("saas", { hide_integrations_page: true });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          result.current.find((item) => item.to === "/settings/integrations"),
        ).toBeUndefined();
        // Other pages should still be present
        expect(
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/billing"),
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
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeUndefined();
        expect(
          result.current.find((item) => item.to === "/settings/billing"),
        ).toBeUndefined();
        expect(
          result.current.find((item) => item.to === "/settings/integrations"),
        ).toBeUndefined();
        // Non-hidden pages should still be present
        expect(
          result.current.find((item) => item.to === "/settings"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/app"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/secrets"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/mcp"),
        ).toBeDefined();
      });
    });

    it("should keep all pages visible when no hide flags are set", async () => {
      mockConfigWithFeatureFlags("saas", {});
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        // All SAAS pages should be present
        expect(
          result.current.find((item) => item.to === "/settings/user"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/billing"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/integrations"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/app"),
        ).toBeDefined();
      });
    });

    it("should filter out '/settings/integrations' in OSS mode when hide_integrations_page is true", async () => {
      mockConfigWithFeatureFlags("oss", { hide_integrations_page: true });
      const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

      await waitFor(() => {
        expect(
          result.current.find((item) => item.to === "/settings/integrations"),
        ).toBeUndefined();
        // Other OSS pages should still be present
        expect(
          result.current.find((item) => item.to === "/settings"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/mcp"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/app"),
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
          result.current.find((item) => item.to === "/settings"),
        ).toBeUndefined();
        expect(
          result.current.find((item) => item.to === "/settings/integrations"),
        ).toBeUndefined();
        // Other OSS pages should still be present
        expect(
          result.current.find((item) => item.to === "/settings/mcp"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/app"),
        ).toBeDefined();
        expect(
          result.current.find((item) => item.to === "/settings/secrets"),
        ).toBeDefined();
      });
    });
  });
});
