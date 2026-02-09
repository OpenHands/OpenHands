import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useOrgTypeAndAccess } from "#/hooks/use-org-type-and-access";

// Mock the dependencies
vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: vi.fn(),
}));

vi.mock("#/hooks/query/use-organizations", () => ({
  useOrganizations: vi.fn(),
}));

// Import mocked modules
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";

const mockUseSelectedOrganizationId = vi.mocked(useSelectedOrganizationId);
const mockUseOrganizations = vi.mocked(useOrganizations);

const queryClient = new QueryClient();
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe("useOrgTypeAndAccess", () => {
  beforeEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("should return false for all booleans when no organization is selected", async () => {
    mockUseSelectedOrganizationId.mockReturnValue({
      organizationId: null,
      setOrganizationId: vi.fn(),
    });
    mockUseOrganizations.mockReturnValue({
      data: [],
    } as unknown as ReturnType<typeof useOrganizations>);

    const { result } = renderHook(() => useOrgTypeAndAccess(), { wrapper });

    await waitFor(() => {
      expect(result.current.selectedOrg).toBeUndefined();
      expect(result.current.isPersonalOrg).toBe(false);
      expect(result.current.isTeamOrg).toBe(false);
      expect(result.current.canViewOrgRoutes).toBe(false);
      expect(result.current.organizationId).toBeNull();
    });
  });

  it("should return isPersonalOrg=true and isTeamOrg=false for personal org", async () => {
    const personalOrg = { id: "org-1", is_personal: true, name: "Personal" };
    mockUseSelectedOrganizationId.mockReturnValue({
      organizationId: "org-1",
      setOrganizationId: vi.fn(),
    });
    mockUseOrganizations.mockReturnValue({
      data: [personalOrg],
    } as unknown as ReturnType<typeof useOrganizations>);

    const { result } = renderHook(() => useOrgTypeAndAccess(), { wrapper });

    await waitFor(() => {
      expect(result.current.selectedOrg).toEqual(personalOrg);
      expect(result.current.isPersonalOrg).toBe(true);
      expect(result.current.isTeamOrg).toBe(false);
      expect(result.current.canViewOrgRoutes).toBe(false);
      expect(result.current.organizationId).toBe("org-1");
    });
  });

  it("should return isPersonalOrg=false and isTeamOrg=true for team org", async () => {
    const teamOrg = { id: "org-2", is_personal: false, name: "Team" };
    mockUseSelectedOrganizationId.mockReturnValue({
      organizationId: "org-2",
      setOrganizationId: vi.fn(),
    });
    mockUseOrganizations.mockReturnValue({
      data: [teamOrg],
    } as unknown as ReturnType<typeof useOrganizations>);

    const { result } = renderHook(() => useOrgTypeAndAccess(), { wrapper });

    await waitFor(() => {
      expect(result.current.selectedOrg).toEqual(teamOrg);
      expect(result.current.isPersonalOrg).toBe(false);
      expect(result.current.isTeamOrg).toBe(true);
      expect(result.current.canViewOrgRoutes).toBe(true);
      expect(result.current.organizationId).toBe("org-2");
    });
  });

  it("should return canViewOrgRoutes=true only when isTeamOrg AND organizationId is truthy", async () => {
    const teamOrg = { id: "org-3", is_personal: false, name: "Team" };
    mockUseSelectedOrganizationId.mockReturnValue({
      organizationId: "org-3",
      setOrganizationId: vi.fn(),
    });
    mockUseOrganizations.mockReturnValue({
      data: [teamOrg],
    } as unknown as ReturnType<typeof useOrganizations>);

    const { result } = renderHook(() => useOrgTypeAndAccess(), { wrapper });

    await waitFor(() => {
      expect(result.current.isTeamOrg).toBe(true);
      expect(result.current.organizationId).toBe("org-3");
      expect(result.current.canViewOrgRoutes).toBe(true);
    });
  });

  it("should treat undefined is_personal field as team org", async () => {
    // Organization without is_personal field (undefined)
    const orgWithoutPersonalField = { id: "org-4", name: "Unknown Type" };
    mockUseSelectedOrganizationId.mockReturnValue({
      organizationId: "org-4",
      setOrganizationId: vi.fn(),
    });
    mockUseOrganizations.mockReturnValue({
      data: [orgWithoutPersonalField],
    } as unknown as ReturnType<typeof useOrganizations>);

    const { result } = renderHook(() => useOrgTypeAndAccess(), { wrapper });

    await waitFor(() => {
      expect(result.current.selectedOrg).toEqual(orgWithoutPersonalField);
      expect(result.current.isPersonalOrg).toBe(false);
      expect(result.current.isTeamOrg).toBe(true);
      expect(result.current.canViewOrgRoutes).toBe(true);
    });
  });
});
