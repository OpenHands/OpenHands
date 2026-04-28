import { describe, expect, it, vi, beforeEach } from "vitest";
import { PermissionKey } from "#/utils/org/permissions";
import { OrganizationMember, OrganizationsQueryData } from "#/types/org";
import {
  getAvailableRolesAUserCanAssign,
  getActiveOrganizationUser,
} from "#/utils/org/permission-checks";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { queryClient } from "#/query-client-config";

// Mock dependencies
vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getMe: vi.fn(),
    getOrganizations: vi.fn(),
  },
}));

vi.mock("#/stores/selected-organization-store", () => ({
  getSelectedOrganizationIdFromStore: vi.fn(),
}));

vi.mock("#/query-client-config", () => ({
  queryClient: {
    getQueryData: vi.fn(),
    fetchQuery: vi.fn(),
    setQueryData: vi.fn(),
  },
}));

// Test fixtures
const mockUser: OrganizationMember = {
  org_id: "org-1",
  user_id: "user-1",
  email: "test@example.com",
  role: "admin",
  llm_api_key: "",
  max_iterations: 100,
  llm_model: "gpt-4",
  llm_base_url: "",
  status: "active",
};

const mockOrganizationsData: OrganizationsQueryData = {
  items: [
    { id: "org-1", name: "Org 1" },
    { id: "org-2", name: "Org 2" },
  ] as OrganizationsQueryData["items"],
  currentOrgId: "org-1",
};

describe("getAvailableRolesAUserCanAssign", () => {
  it("returns empty array if user has no permissions", () => {
    const result = getAvailableRolesAUserCanAssign([]);
    expect(result).toEqual([]);
  });

  it("returns only roles the user has permission for", () => {
    const userPermissions: PermissionKey[] = [
      "change_user_role:member",
      "change_user_role:admin",
    ];
    const result = getAvailableRolesAUserCanAssign(userPermissions);
    expect(result.sort()).toEqual(["admin", "member"].sort());
  });

  it("returns all roles if user has all permissions", () => {
    const allPermissions: PermissionKey[] = [
      "change_user_role:member",
      "change_user_role:admin",
      "change_user_role:owner",
    ];
    const result = getAvailableRolesAUserCanAssign(allPermissions);
    expect(result.sort()).toEqual(["member", "admin", "owner"].sort());
  });
});

describe("getActiveOrganizationUser", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("when orgId exists in store", () => {
    it("should fetch user directly using stored orgId", async () => {
      // Arrange
      vi.mocked(getSelectedOrganizationIdFromStore).mockReturnValue("org-1");
      vi.mocked(queryClient.fetchQuery).mockResolvedValue(mockUser);

      // Act
      const result = await getActiveOrganizationUser();

      // Assert
      expect(result).toEqual(mockUser);
      expect(queryClient.getQueryData).not.toHaveBeenCalled();
      expect(queryClient.fetchQuery).toHaveBeenCalledWith(
        expect.objectContaining({
          queryKey: ["organizations", "org-1", "me"],
        }),
      );
    });

    it("should return undefined when user fetch fails", async () => {
      // Arrange
      vi.mocked(getSelectedOrganizationIdFromStore).mockReturnValue("org-1");
      vi.mocked(queryClient.fetchQuery).mockRejectedValue(
        new Error("Network error"),
      );

      // Act
      const result = await getActiveOrganizationUser();

      // Assert
      expect(result).toBeUndefined();
    });
  });

  describe("when orgId is null in store (page refresh scenario)", () => {
    beforeEach(() => {
      vi.mocked(getSelectedOrganizationIdFromStore).mockReturnValue(null);
    });

    it("should use currentOrgId from cached organizations data", async () => {
      // Arrange
      vi.mocked(queryClient.getQueryData).mockReturnValue(
        mockOrganizationsData,
      );
      vi.mocked(queryClient.fetchQuery).mockResolvedValue(mockUser);

      // Act
      const result = await getActiveOrganizationUser();

      // Assert
      expect(result).toEqual(mockUser);
      expect(queryClient.getQueryData).toHaveBeenCalledWith(["organizations"]);
      expect(queryClient.fetchQuery).toHaveBeenCalledWith(
        expect.objectContaining({
          queryKey: ["organizations", "org-1", "me"],
        }),
      );
    });

    it("should fallback to first org when currentOrgId is null", async () => {
      // Arrange
      const dataWithoutCurrentOrg: OrganizationsQueryData = {
        items: [
          { id: "first-org" },
          { id: "second-org" },
        ] as OrganizationsQueryData["items"],
        currentOrgId: null,
      };
      vi.mocked(queryClient.getQueryData).mockReturnValue(
        dataWithoutCurrentOrg,
      );
      vi.mocked(queryClient.fetchQuery).mockResolvedValue(mockUser);

      // Act
      const result = await getActiveOrganizationUser();

      // Assert
      expect(result).toEqual(mockUser);
      expect(queryClient.fetchQuery).toHaveBeenCalledWith(
        expect.objectContaining({
          queryKey: ["organizations", "first-org", "me"],
        }),
      );
    });

    it("should fetch organizations when not in cache", async () => {
      // Arrange
      vi.mocked(queryClient.getQueryData).mockReturnValue(undefined);
      vi.mocked(queryClient.fetchQuery)
        .mockResolvedValueOnce(mockOrganizationsData) // First call: fetch organizations
        .mockResolvedValueOnce(mockUser); // Second call: fetch user

      // Act
      const result = await getActiveOrganizationUser();

      // Assert
      expect(result).toEqual(mockUser);
      expect(queryClient.fetchQuery).toHaveBeenCalledTimes(2);
      expect(queryClient.fetchQuery).toHaveBeenNthCalledWith(
        1,
        expect.objectContaining({
          queryKey: ["organizations"],
        }),
      );
      expect(queryClient.fetchQuery).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          queryKey: ["organizations", "org-1", "me"],
        }),
      );
    });

    it("should return undefined when fetching organizations fails", async () => {
      // Arrange
      vi.mocked(queryClient.getQueryData).mockReturnValue(undefined);
      vi.mocked(queryClient.fetchQuery).mockRejectedValue(
        new Error("Failed to fetch organizations"),
      );

      // Act
      const result = await getActiveOrganizationUser();

      // Assert
      expect(result).toBeUndefined();
    });

    it("should return undefined when no organizations exist", async () => {
      // Arrange
      const emptyData: OrganizationsQueryData = {
        items: [],
        currentOrgId: null,
      };
      vi.mocked(queryClient.getQueryData).mockReturnValue(emptyData);

      // Act
      const result = await getActiveOrganizationUser();

      // Assert
      expect(result).toBeUndefined();
      // Should not attempt to fetch user since there's no orgId
      expect(queryClient.fetchQuery).not.toHaveBeenCalled();
    });
  });
});
