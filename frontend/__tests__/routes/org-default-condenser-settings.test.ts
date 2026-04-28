import { describe, expect, it, vi, beforeEach } from "vitest";
import { redirect } from "react-router";

/**
 * Regression test: when getActiveOrganizationUser() returns undefined on
 * /settings/org-defaults/condenser, the loader must redirect to a safe
 * terminal page (e.g. /settings/user) — NOT to /settings/condenser,
 * which unconditionally bounces back in SaaS mode and creates an
 * infinite redirect loop that crashes the browser tab.
 */

vi.mock("react-router", () => ({
  redirect: vi.fn((path: string) => ({ _tag: "redirect", path })),
}));

vi.mock("#/utils/org/permission-checks", () => ({
  getActiveOrganizationUser: vi.fn(),
}));

vi.mock("#/api/option-service/option-service.api", () => ({
  default: {
    getConfig: vi.fn().mockResolvedValue({
      app_mode: "saas",
      feature_flags: {
        hide_users_page: false,
        hide_billing_page: false,
        hide_integrations_page: false,
        hide_llm_settings: false,
      },
    }),
  },
}));

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getOrganizations: vi.fn().mockResolvedValue({
      items: [{ id: "org-1", name: "Team Org", is_personal: false }],
      currentOrgId: "org-1",
    }),
  },
}));

vi.mock("#/stores/selected-organization-store", () => ({
  getSelectedOrganizationIdFromStore: vi.fn(() => "org-1"),
}));

vi.mock("#/query-client-config", () => ({
  queryClient: {
    getQueryData: vi.fn(() => ({
      app_mode: "saas",
      feature_flags: {
        hide_users_page: false,
        hide_billing_page: false,
        hide_integrations_page: false,
        hide_llm_settings: false,
      },
    })),
    setQueryData: vi.fn(),
    fetchQuery: vi.fn(({ queryKey }: { queryKey: unknown[] }) => {
      if (queryKey[0] === "organizations") {
        return Promise.resolve({
          items: [{ id: "org-1", name: "Team Org", is_personal: false }],
          currentOrgId: "org-1",
        });
      }
      return Promise.resolve({
        app_mode: "saas",
        feature_flags: {
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: false,
          hide_llm_settings: false,
        },
      });
    }),
  },
}));

import { getActiveOrganizationUser } from "#/utils/org/permission-checks";
import { clientLoader } from "#/routes/org-default-condenser-settings";

beforeEach(() => {
  vi.clearAllMocks();
});

describe("org-defaults/condenser clientLoader", () => {
  it("redirects to a safe terminal page when user is undefined", async () => {
    vi.mocked(getActiveOrganizationUser).mockResolvedValue(undefined);

    await clientLoader({
      request: new Request(
        "http://localhost/settings/org-defaults/condenser",
      ),
    });

    expect(redirect).toHaveBeenCalledWith("/settings/user");
  });
});
