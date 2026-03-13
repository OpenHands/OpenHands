import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useOrganizations } from "#/hooks/query/use-organizations";
import type { Organization } from "#/types/org";

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getOrganizations: vi.fn(),
  },
}));

// Mock useIsAuthed to return authenticated
vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

// Mock useConfig to return SaaS mode (organizations are a SaaS-only feature)
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: { app_mode: "saas" } }),
}));

const mockGetOrganizations = vi.mocked(organizationService.getOrganizations);

function createMinimalOrg(
  id: string,
  name: string,
  is_personal?: boolean,
): Organization {
  return {
    id,
    name,
    is_personal,
    contact_name: "",
    contact_email: "",
    conversation_expiration: 0,
    agent: "",
    default_max_iterations: 0,
    security_analyzer: "",
    confirmation_mode: false,
    default_llm_model: "",
    default_llm_api_key_for_byor: "",
    default_llm_base_url: "",
    remote_runtime_resource_factor: 0,
    enable_default_condenser: false,
    billing_margin: 0,
    enable_proactive_conversation_starters: false,
    sandbox_base_container_image: "",
    sandbox_runtime_container_image: "",
    org_version: 0,
    mcp_config: { tools: [], settings: {} },
    search_api_key: null,
    sandbox_api_key: null,
    max_budget_per_task: 0,
    enable_solvability_analysis: false,
    v1_enabled: false,
    credits: 0,
  };
}

describe("useOrganizations", () => {
  let queryClient: QueryClient;

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    vi.clearAllMocks();
  });

  it("sorts personal workspace first, then non-personal alphabetically by name", async () => {
    // API returns unsorted: Beta, Personal, Acme, All Hands
    mockGetOrganizations.mockResolvedValue({
      items: [
        createMinimalOrg("3", "Beta LLC", false),
        createMinimalOrg("1", "Personal Workspace", true),
        createMinimalOrg("2", "Acme Corp", false),
        createMinimalOrg("4", "All Hands AI", false),
      ],
      currentOrgId: "1",
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const { organizations } = result.current.data!;
    expect(organizations).toHaveLength(4);
    expect(organizations[0].id).toBe("1");
    expect(organizations[0].is_personal).toBe(true);
    expect(organizations[0].name).toBe("Personal Workspace");
    expect(organizations[1].name).toBe("Acme Corp");
    expect(organizations[2].name).toBe("All Hands AI");
    expect(organizations[3].name).toBe("Beta LLC");
  });

  it("treats missing is_personal as false and sorts by name", async () => {
    mockGetOrganizations.mockResolvedValue({
      items: [
        createMinimalOrg("1", "Zebra Org"), // no is_personal
        createMinimalOrg("2", "Alpha Org", true), // personal first
        createMinimalOrg("3", "Mango Org"), // no is_personal
      ],
      currentOrgId: "2",
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const { organizations } = result.current.data!;
    expect(organizations[0].id).toBe("2");
    expect(organizations[0].is_personal).toBe(true);
    expect(organizations[1].name).toBe("Mango Org");
    expect(organizations[2].name).toBe("Zebra Org");
  });

  it("handles missing name by treating as empty string for sort", async () => {
    const orgWithName = createMinimalOrg("2", "Beta", false);
    const orgNoName = { ...createMinimalOrg("1", "Alpha", false) };
    delete (orgNoName as Record<string, unknown>).name;
    mockGetOrganizations.mockResolvedValue({
      items: [orgWithName, orgNoName] as Organization[],
      currentOrgId: "1",
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const { organizations } = result.current.data!;
    // undefined name is coerced to ""; "" sorts before "Beta"
    expect(organizations[0].id).toBe("1");
    expect(organizations[1].id).toBe("2");
    expect(organizations[1].name).toBe("Beta");
  });

  it("does not mutate the original array from the API", async () => {
    const apiOrgs = [
      createMinimalOrg("2", "Acme", false),
      createMinimalOrg("1", "Personal", true),
    ];
    mockGetOrganizations.mockResolvedValue({
      items: apiOrgs,
      currentOrgId: "1",
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Hook sorts a copy ([...data]), so API order unchanged
    expect(apiOrgs[0].id).toBe("2");
    expect(apiOrgs[1].id).toBe("1");
    // Returned data is sorted
    expect(result.current.data!.organizations[0].id).toBe("1");
    expect(result.current.data!.organizations[1].id).toBe("2");
  });
});
