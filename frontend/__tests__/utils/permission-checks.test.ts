import { describe, expect, it, vi, beforeEach } from "vitest";
import { PermissionKey } from "#/utils/org/permissions";

// Mock dependencies for getActiveOrganizationUser tests
vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getMe: vi.fn(),
  },
}));

vi.mock("#/stores/selected-organization-store", () => ({
  getSelectedOrganizationIdFromStore: vi.fn(),
}));

vi.mock("#/utils/query-client-getters", () => ({
  getMeFromQueryClient: vi.fn(),
}));

vi.mock("#/query-client-config", () => ({
  queryClient: {
    setQueryData: vi.fn(),
  },
}));

// Import after mocks are set up
import {
  getAvailableRolesAUserCanAssign,
  getActiveOrganizationUser,
} from "#/utils/org/permission-checks";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { getMeFromQueryClient } from "#/utils/query-client-getters";

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

  it("should return undefined when API call throws an error", async () => {
    // Arrange: orgId exists, cache is empty, API call fails
    vi.mocked(getSelectedOrganizationIdFromStore).mockReturnValue("org-1");
    vi.mocked(getMeFromQueryClient).mockReturnValue(undefined);
    vi.mocked(organizationService.getMe).mockRejectedValue(
      new Error("Network error"),
    );

    // Act
    const result = await getActiveOrganizationUser();

    // Assert: should return undefined instead of propagating the error
    expect(result).toBeUndefined();
  });
});
