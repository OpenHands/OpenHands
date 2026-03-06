import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import OptionService from "#/api/option-service/option-service.api";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { WebClientFeatureFlags } from "#/api/option-service/option.types";

const queryClient = new QueryClient();
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

const mockConfig = (appMode: "saas" | "oss", hideLlmSettings = false) => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue({
    app_mode: appMode,
    feature_flags: { hide_llm_settings: hideLlmSettings },
  } as Awaited<ReturnType<typeof OptionService.getConfig>>);
};

const mockConfigWithFeatureFlags = (
  appMode: "saas" | "oss",
  featureFlags: Partial<WebClientFeatureFlags>,
) => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue({
    app_mode: appMode,
    feature_flags: {
      enable_billing: false,
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
  });

  it("should return SAAS_NAV_ITEMS when app_mode is 'saas'", async () => {
    mockConfig("saas");
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(result.current).toEqual(SAAS_NAV_ITEMS);
    });
  });

  it("should return OSS_NAV_ITEMS when app_mode is 'oss'", async () => {
    mockConfig("oss");
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(result.current).toEqual(OSS_NAV_ITEMS);
    });
  });

  it("should filter out '/settings' item when hide_llm_settings feature flag is enabled", async () => {
    mockConfig("saas", true);
    const { result } = renderHook(() => useSettingsNavItems(), { wrapper });

    await waitFor(() => {
      expect(
        result.current.find((item) => item.to === "/settings"),
      ).toBeUndefined();
    });
  });

  describe("hide page feature flags", () => {
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
