import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { redirect } from "react-router";

// Mock dependencies before importing the module under test
vi.mock("react-router", () => ({
  redirect: vi.fn((path: string) => ({ type: "redirect", path })),
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

const mockConfig = {
  app_mode: "saas",
  feature_flags: {
    hide_users_page: false,
    hide_billing_page: false,
    hide_integrations_page: false,
    hide_llm_settings: false,
  },
};

vi.mock("#/query-client-config", () => ({
  queryClient: {
    getQueryData: vi.fn(() => mockConfig),
    setQueryData: vi.fn(),
    fetchQuery: vi.fn(() => Promise.resolve(mockConfig)),
  },
}));

// Import after mocks are set up
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { getActiveOrganizationUser } from "#/utils/org/permission-checks";

// Helper to create a mock request
const createMockRequest = (pathname: string = "/settings/billing") => ({
  request: new Request(`http://localhost${pathname}`),
});

describe("createPermissionGuard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe("permission checking", () => {
    it("should redirect when user lacks required permission", async () => {
      // Arrange: member lacks view_billing permission
      vi.mocked(getActiveOrganizationUser).mockResolvedValue({
        org_id: "org-1",
        user_id: "user-1",
        email: "test@example.com",
        role: "member",
        llm_api_key: "",
        max_iterations: 100,
        llm_model: "gpt-4",
        llm_base_url: "",
        status: "active",
      });

      // Act
      const guard = createPermissionGuard("view_billing");
      await guard(createMockRequest("/settings/billing"));

      // Assert: should redirect to first available path (/settings/user in SaaS mode)
      expect(redirect).toHaveBeenCalledWith("/settings/user");
    });

    it("should allow access when user has required permission", async () => {
      // Arrange: admin has view_billing permission
      vi.mocked(getActiveOrganizationUser).mockResolvedValue({
        org_id: "org-1",
        user_id: "user-1",
        email: "admin@example.com",
        role: "admin",
        llm_api_key: "",
        max_iterations: 100,
        llm_model: "gpt-4",
        llm_base_url: "",
        status: "active",
      });

      // Act
      const guard = createPermissionGuard("view_billing");
      const result = await guard(createMockRequest("/settings/billing"));

      // Assert: should not redirect, return null
      expect(redirect).not.toHaveBeenCalled();
      expect(result).toBeNull();
    });

    it("should redirect when user is undefined (no org selected)", async () => {
      // Arrange: no user (e.g., no organization selected)
      vi.mocked(getActiveOrganizationUser).mockResolvedValue(undefined);

      // Act
      const guard = createPermissionGuard("view_billing");
      await guard(createMockRequest("/settings/billing"));

      // Assert: should redirect to first available path
      expect(redirect).toHaveBeenCalledWith("/settings/user");
    });

    it("should redirect when user is undefined even for member-level permissions", async () => {
      // Arrange: no user — manage_secrets is a member-level permission,
      // but undefined user should NOT get member access
      vi.mocked(getActiveOrganizationUser).mockResolvedValue(undefined);

      // Act
      const guard = createPermissionGuard("manage_secrets");
      await guard(createMockRequest("/settings/secrets"));

      // Assert: should redirect, not silently grant member-level access
      expect(redirect).toHaveBeenCalledWith("/settings/user");
    });
  });

  describe("custom redirect path", () => {
    it("should redirect to custom path when specified", async () => {
      // Arrange: member lacks permission
      vi.mocked(getActiveOrganizationUser).mockResolvedValue({
        org_id: "org-1",
        user_id: "user-1",
        email: "test@example.com",
        role: "member",
        llm_api_key: "",
        max_iterations: 100,
        llm_model: "gpt-4",
        llm_base_url: "",
        status: "active",
      });

      // Act
      const guard = createPermissionGuard("view_billing", "/custom/redirect");
      await guard(createMockRequest("/settings/billing"));

      // Assert: should redirect to custom path
      expect(redirect).toHaveBeenCalledWith("/custom/redirect");
    });
  });

  describe("infinite loop prevention", () => {
    it("should return null instead of redirecting when fallback path equals current path", async () => {
      // Arrange: no user
      vi.mocked(getActiveOrganizationUser).mockResolvedValue(undefined);

      // Act: access /settings/user when fallback would also be /settings/user
      const guard = createPermissionGuard("view_billing");
      const result = await guard(createMockRequest("/settings/user"));

      // Assert: should NOT redirect to avoid infinite loop
      expect(redirect).not.toHaveBeenCalled();
      expect(result).toBeNull();
    });
  });
});
