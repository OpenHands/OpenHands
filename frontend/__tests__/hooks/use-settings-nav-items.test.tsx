import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import OptionService from "#/api/option-service/option-service.api";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";

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

const mockConfig = (appMode: "saas" | "oss", hideLlmSettings = false) => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue({
    APP_MODE: appMode,
    FEATURE_FLAGS: { HIDE_LLM_SETTINGS: hideLlmSettings },
  } as Awaited<ReturnType<typeof OptionService.getConfig>>);
};

describe("useSettingsNavItems", () => {
  beforeEach(() => {
    queryClient.clear();
    // Reset mocks to a default state where org routes are visible but not billing
    // (team org, admin role, org selected)
    mockOrgTypeAndAccess.isPersonalOrg = false;
    mockOrgTypeAndAccess.isTeamOrg = true;
    mockOrgTypeAndAccess.organizationId = "org-123";
    mockMe.data = { role: "admin" };
  });

  it("should return SAAS_NAV_ITEMS (without billing) when APP_MODE is 'saas' and team org", async () => {
    mockConfig("saas");
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      // Team org: all items except billing
      const expectedItems = SAAS_NAV_ITEMS.filter(
        (item) => item.to !== "/settings/billing",
      );
      expect(result.current).toEqual(expectedItems);
    });
  });

  it("should return all SAAS_NAV_ITEMS when APP_MODE is 'saas' and personal org", async () => {
    mockConfig("saas");
    mockOrgTypeAndAccess.isPersonalOrg = true;
    mockOrgTypeAndAccess.isTeamOrg = false;

    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      // Personal org: all items except org routes (personal orgs can't have org settings)
      const expectedItems = SAAS_NAV_ITEMS.filter(
        (item) =>
          item.to !== "/settings/org" && item.to !== "/settings/org-members",
      );
      expect(result.current).toEqual(expectedItems);
    });
  });

  it("should return OSS_NAV_ITEMS when APP_MODE is 'oss'", async () => {
    mockConfig("oss");
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(result.current).toEqual(OSS_NAV_ITEMS);
    });
  });

  it("should filter out '/settings' item when HIDE_LLM_SETTINGS feature flag is enabled", async () => {
    mockConfig("saas", true);
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(
        result.current.find((item) => item.to === "/settings"),
      ).toBeUndefined();
    });
  });

  describe("org-type and role-based filtering", () => {
    // NOTE: Each test sets up its specific mock state as needed
    // Default state (team org, admin, org selected) is set in the outer beforeEach

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
});
