import { describe, expect, it, vi, beforeEach } from "vitest";

vi.mock("react-router", () => ({
  redirect: vi.fn((path: string) => ({ type: "redirect", path })),
}));

const mockConfig = {
  app_mode: "saas" as "saas" | "oss",
  feature_flags: {},
};

vi.mock("#/api/option-service/option-service.api", () => ({
  default: { getConfig: vi.fn(async () => mockConfig) },
}));

const mockOrganizations = {
  items: [
    { id: "personal-org", name: "Personal", is_personal: true },
    { id: "team-org", name: "Acme", is_personal: false },
  ],
  currentOrgId: "team-org" as string | null,
};

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getOrganizations: vi.fn(async () => mockOrganizations),
  },
}));

let storeOrgId: string | null = null;
vi.mock("#/stores/selected-organization-store", () => ({
  getSelectedOrganizationIdFromStore: () => storeOrgId,
}));

vi.mock("#/query-client-config", () => ({
  queryClient: {
    fetchQuery: vi.fn(async (opts: { queryFn: () => Promise<unknown> }) =>
      opts.queryFn(),
    ),
  },
}));

import { redirect } from "react-router";
import { requireOrgDefaultsRedirect as requirePersonalWorkspaceLoader } from "#/utils/org/saas-redirect-to-org-defaults-guard";

const createRequest = (pathname: string) => ({
  request: new Request(`http://localhost${pathname}`),
});

describe("requirePersonalWorkspaceLoader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockConfig.app_mode = "saas";
    mockOrganizations.currentOrgId = "team-org";
    storeOrgId = null;
  });

  it("redirects to the org-defaults equivalent when the active org is a team workspace", async () => {
    storeOrgId = "team-org";
    const guard = requirePersonalWorkspaceLoader("/settings/org-defaults");

    const result = await guard(createRequest("/settings"));

    expect(redirect).toHaveBeenCalledWith("/settings/org-defaults");
    expect(result).toEqual({ type: "redirect", path: "/settings/org-defaults" });
  });

  it("redirects to the org-defaults equivalent even when the active org is the personal workspace", async () => {
    storeOrgId = "personal-org";
    const guard = requirePersonalWorkspaceLoader("/settings/org-defaults");

    const result = await guard(createRequest("/settings"));

    expect(redirect).toHaveBeenCalledWith("/settings/org-defaults");
    expect(result).toEqual({ type: "redirect", path: "/settings/org-defaults" });
  });

  it("skips the guard entirely in OSS mode", async () => {
    mockConfig.app_mode = "oss";
    storeOrgId = "team-org";
    const guard = requirePersonalWorkspaceLoader("/settings/org-defaults");

    const result = await guard(createRequest("/settings"));

    expect(result).toBeNull();
    expect(redirect).not.toHaveBeenCalled();
  });
});
