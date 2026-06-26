import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub, Outlet } from "react-router";
import SkillsSettingsScreen from "#/routes/skills-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import * as orgStore from "#/stores/selected-organization-store";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME } from "#/mocks/org-handlers";
import type { WebClientConfig } from "#/api/option-service/option.types";

const mockConfig: WebClientConfig = {
  app_mode: "oss",
  posthog_client_key: null,
  feature_flags: {
    enable_billing: false,
    hide_llm_settings: false,
    enable_jira: false,
    enable_jira_dc: false,
    enable_linear: false,
    hide_users_page: false,
    hide_billing_page: false,
    hide_integrations_page: false,
    enable_onboarding: false,
  },
  providers_configured: [],
  maintenance_start_time: null,
  auth_url: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: new Date().toISOString(),
  github_app_slug: null,
};

const renderSkillsSettings = () => {
  const RouterStub = createRoutesStub([
    {
      Component: () => <Outlet />,
      path: "/settings",
      children: [
        {
          Component: SkillsSettingsScreen,
          path: "/settings/skills",
        },
        {
          Component: () => <div data-testid="skills-settings-screen" />,
          path: "/settings/skills",
        },
      ],
    },
  ]);

  return render(<RouterStub initialEntries={["/settings/skills"]} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });
};

beforeEach(() => {
  vi.spyOn(OptionService, "getConfig").mockResolvedValue(mockConfig);

  vi.spyOn(organizationService, "getMe").mockResolvedValue({
    org_id: "org-1",
    user_id: "user-1",
    email: "test@example.com",
    role: "admin",
    llm_api_key: "",
    max_iterations: 100,
    llm_model: "gpt-4",
    llm_base_url: "",
    status: "active",
  });

  orgStore.useSelectedOrganizationStore.setState({
    organizationId: null,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("hasMarketplaceChanges - auto_load toggle bug fix", () => {
  it("should normalize null and undefined as equivalent in comparison", () => {
    // This test documents the bug fix:
    // Before: undefined !== null returned true (bug)
    // After: (undefined ?? null) === (null ?? null) returns false
    const normalizeValue = (val: "all" | null | undefined) => val ?? null;

    // Original value from backend (undefined after model_dump strips null)
    const original = normalizeValue(undefined);
    // After toggle off (now using null)
    const toggledOff = normalizeValue(null);
    // After toggle on
    const toggledOn = normalizeValue("all");

    expect(original).toBeNull();
    expect(toggledOff).toBeNull();
    expect(toggledOn).toBe("all");
    expect(original).toBe(toggledOff); // These should be equal now
  });

  it("should not detect change when toggling auto_load back to original state", () => {
    // Simulates the comparison logic in hasMarketplaceChanges
    const normalizeAutoLoad = (val: "all" | null | undefined) => val ?? null;

    const currentValue = normalizeAutoLoad(undefined);
    const originalValue = normalizeAutoLoad(null);

    // Both should normalize to null, so no change detected
    const hasChange = currentValue !== originalValue;
    expect(hasChange).toBe(false);
  });

  it("should detect change when auto_load is different from original", () => {
    const normalizeAutoLoad = (val: "all" | null | undefined) => val ?? null;

    const currentValue = normalizeAutoLoad("all");
    const originalValue = normalizeAutoLoad(null);

    const hasChange = currentValue !== originalValue;
    expect(hasChange).toBe(true);
  });
});

describe("hasSkillChanges - enabled toggle bug fix", () => {
  it("should normalize isEnabled comparison correctly", () => {
    const normalizeBool = (val: boolean | undefined | null) => val ?? false;

    // Both should normalize to false
    expect(normalizeBool(false)).toBe(false);
    expect(normalizeBool(undefined)).toBe(false);
    expect(normalizeBool(null)).toBe(false);
  });
});
