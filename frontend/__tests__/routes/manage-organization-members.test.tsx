import { describe, expect, it, vi, test, beforeEach, afterEach } from "vitest";
import { render, screen, within, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import { selectOrganization } from "test-utils";
import { organizationService } from "#/api/organization-service/organization-service.api";
import ManageOrganizationMembers from "#/routes/manage-organization-members";
import SettingsScreen, {
  clientLoader as settingsClientLoader,
} from "#/routes/settings";
import {
  ORGS_AND_MEMBERS,
  resetOrgMockData,
  resetOrgsAndMembersMockData,
} from "#/mocks/org-handlers";
import OptionService from "#/api/option-service/option-service.api";

function ManageOrganizationMembersWithPortalRoot() {
  return (
    <div>
      <ManageOrganizationMembers />
      <div data-testid="portal-root" id="portal-root" />
    </div>
  );
}

const RouteStub = createRoutesStub([
  {
    // @ts-expect-error - ignoreing error for test stub
    loader: settingsClientLoader,
    Component: SettingsScreen,
    path: "/settings",
    HydrateFallback: () => <div>Loading...</div>,
    children: [
      {
        Component: ManageOrganizationMembersWithPortalRoot,
        path: "/settings/org-members",
      },
      {
        Component: () => <div data-testid="user-settings" />,
        path: "/settings/user",
      },
    ],
  },
]);

let queryClient: QueryClient;

describe("Manage Organization Members Route", () => {
  const getMeSpy = vi.spyOn(organizationService, "getMe");

  beforeEach(() => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    // @ts-expect-error - only return APP_MODE for these tests
    getConfigSpy.mockResolvedValue({
      APP_MODE: "saas",
    });

    queryClient = new QueryClient();

    // Set default mock for user (admin role has invite permission)
    getMeSpy.mockResolvedValue({
      id: "1",
      email: "test@example.com",
      role: "admin",
      status: "active",
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // Reset organization mock data to ensure clean state between tests
    resetOrgMockData();
    // Reset ORGS_AND_MEMBERS to initial state
    resetOrgsAndMembersMockData();
    // Clear queryClient cache to ensure fresh data for next test
    queryClient.clear();
  });

  const renderManageOrganizationMembers = () =>
    render(<RouteStub initialEntries={["/settings/org-members"]} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });

  // Helper function to find a member by email
  const findMemberByEmail = async (email: string) => {
    const memberListItems = await screen.findAllByTestId("member-item");
    const member = memberListItems.find((item) =>
      within(item).queryByText(email),
    );
    if (!member) {
      throw new Error(`Could not find member with email: ${email}`);
    }
    return member;
  };

  // Helper function to open role dropdown for a member
  const openRoleDropdown = async (
    memberElement: HTMLElement,
    roleText: string,
  ) => {
    // Find the role text that's clickable (has cursor-pointer class or is the main role display)
    // Use a more specific query to avoid matching dropdown options
    const roleElement = within(memberElement).getByText(
      new RegExp(`^${roleText}$`, "i"),
    );
    await userEvent.click(roleElement);
    return within(memberElement).getByTestId(
      "organization-member-role-context-menu",
    );
  };

  // Helper function to change member role
  const changeMemberRole = async (
    memberElement: HTMLElement,
    currentRole: string,
    newRole: string,
  ) => {
    const dropdown = await openRoleDropdown(memberElement, currentRole);
    const roleOption = within(dropdown).getByText(new RegExp(newRole, "i"));
    await userEvent.click(roleOption);
  };

  // Helper function to verify dropdown is not visible
  const expectDropdownNotVisible = (memberElement: HTMLElement) => {
    expect(
      within(memberElement).queryByTestId(
        "organization-member-role-context-menu",
      ),
    ).not.toBeInTheDocument();
  };

  // Helper function to setup test with user and organization
  const setupTestWithUserAndOrg = async (
    userData: {
      id: string;
      email: string;
      role: "owner" | "admin" | "user";
      status: "active" | "invited";
    },
    orgIndex: number,
  ) => {
    getMeSpy.mockResolvedValue(userData);
    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");
    await selectOrganization({ orgIndex });
  };

  // Helper function to create updateMemberRole spy
  const createUpdateMemberRoleSpy = () =>
    vi.spyOn(organizationService, "updateMemberRole");

  // Helper function to verify role change is not permitted
  const verifyRoleChangeNotPermitted = async (
    userData: {
      id: string;
      email: string;
      role: "owner" | "admin" | "user";
      status: "active" | "invited";
    },
    orgIndex: number,
    targetMemberIndex: number,
    expectedRoleText: string,
  ) => {
    await setupTestWithUserAndOrg(userData, orgIndex);

    const memberListItems = await screen.findAllByTestId("member-item");
    const targetMember = memberListItems[targetMemberIndex];
    const roleText = within(targetMember).getByText(
      new RegExp(`^${expectedRoleText}$`, "i"),
    );
    expect(roleText).toBeInTheDocument();
    await userEvent.click(roleText);

    // Verify that the dropdown does not open
    expectDropdownNotVisible(targetMember);
  };

  // Helper function to setup invite test (render and select organization)
  const setupInviteTest = async (orgIndex: number = 0) => {
    renderManageOrganizationMembers();
    await selectOrganization({ orgIndex });
  };

  // Helper function to setup test with organization (waits for settings screen)
  const setupTestWithOrg = async (orgIndex: number = 0) => {
    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");
    await selectOrganization({ orgIndex });
  };

  // Helper function to find invite button
  const findInviteButton = async () =>
    await screen.findByRole("button", {
      name: /ORG\$INVITE_ORGANIZATION_MEMBER/i,
    });

  // Helper function to verify all three role options are present in dropdown
  const expectAllRoleOptionsPresent = (dropdown: HTMLElement) => {
    expect(within(dropdown).getByText(/owner/i)).toBeInTheDocument();
    expect(within(dropdown).getByText(/admin/i)).toBeInTheDocument();
    expect(within(dropdown).getByText(/user/i)).toBeInTheDocument();
  };

  // Helper function to close dropdown by clicking outside
  const closeDropdown = async () => {
    await userEvent.click(document.body);
  };

  // Helper function to verify owner option is not present in dropdown
  const expectOwnerOptionNotPresent = (dropdown: HTMLElement) => {
    expect(within(dropdown).queryByText(/owner/i)).not.toBeInTheDocument();
  };

  it("should render", async () => {
    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");
  });

  it("should navigate away from the page if not saas", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    // @ts-expect-error - only return APP_MODE for these tests
    getConfigSpy.mockResolvedValue({
      APP_MODE: "oss",
    });

    renderManageOrganizationMembers();
    expect(
      screen.queryByTestId("manage-organization-members-settings"),
    ).not.toBeInTheDocument();
  });

  it("should allow the user to select an organization", async () => {
    const getOrganizationMembersSpy = vi.spyOn(
      organizationService,
      "getOrganizationMembers",
    );

    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");

    expect(getOrganizationMembersSpy).not.toHaveBeenCalled();

    await selectOrganization({ orgIndex: 0 });
    expect(getOrganizationMembersSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
    });
  });

  it("should render the list of organization members", async () => {
    await setupTestWithOrg(0);
    const members = ORGS_AND_MEMBERS["1"];

    const memberListItems = await screen.findAllByTestId("member-item");
    expect(memberListItems).toHaveLength(members.length);

    members.forEach((member) => {
      expect(screen.getByText(member.email)).toBeInTheDocument();
      expect(screen.getByText(member.role)).toBeInTheDocument();
    });
  });

  test("an admin should be able to change the role of a organization member", async () => {
    await setupTestWithUserAndOrg(
      {
        id: "1",
        email: "test@example.com",
        role: "admin",
        status: "active",
      },
      0,
    );

    const updateMemberRoleSpy = createUpdateMemberRoleSpy();

    const memberListItems = await screen.findAllByTestId("member-item");
    const userRoleMember = memberListItems[2]; // third member is "user"

    let userCombobox = within(userRoleMember).getByText(/^User$/i);
    expect(userCombobox).toBeInTheDocument();

    // Change role from user to admin
    await changeMemberRole(userRoleMember, "user", "admin");

    expect(updateMemberRoleSpy).toHaveBeenCalledExactlyOnceWith({
      userId: "3", // assuming the third member is the one being updated
      orgId: "1",
      role: "admin",
    });
    expectDropdownNotVisible(userRoleMember);

    // Verify the role has been updated in the UI
    userCombobox = within(userRoleMember).getByText(/^Admin$/i);
    expect(userCombobox).toBeInTheDocument();

    // Revert the role back to user
    await changeMemberRole(userRoleMember, "admin", "user");

    expect(updateMemberRoleSpy).toHaveBeenNthCalledWith(2, {
      userId: "3",
      orgId: "1",
      role: "user",
    });

    // Verify the role has been reverted in the UI
    userCombobox = within(userRoleMember).getByText(/^User$/i);
    expect(userCombobox).toBeInTheDocument();
  });

  it("should not allow an admin to change the owner's role", async () => {
    await verifyRoleChangeNotPermitted(
      {
        id: "1",
        email: "test@example.com",
        role: "admin",
        status: "active",
      },
      2, // user is admin in org 3
      0, // first member is "owner"
      "Owner",
    );
  });

  it("should not allow an admin to change another admin's role", async () => {
    await verifyRoleChangeNotPermitted(
      {
        id: "1",
        email: "test@example.com",
        role: "admin",
        status: "active",
      },
      2, // user is admin in org 3
      1, // second member is "admin"
      "Admin",
    );
  });

  it("should not allow a user to change their own role", async () => {
    // Mock the /me endpoint to return a user ID that matches one of the members
    await verifyRoleChangeNotPermitted(
      {
        id: "1", // Same as Alice from org 1
        email: "alice@acme.org",
        role: "owner",
        status: "active",
      },
      0,
      0, // First member is Alice (id: "1")
      "Owner",
    );
  });

  it("should show a remove option in the role dropdown and remove the user from the list", async () => {
    const removeMemberSpy = vi.spyOn(organizationService, "removeMember");

    await setupTestWithOrg(0);

    // Get initial member count
    const memberListItems = await screen.findAllByTestId("member-item");
    const initialMemberCount = memberListItems.length;

    const userRoleMember = memberListItems[2]; // third member is "user"
    const userEmail = within(userRoleMember).getByText("charlie@acme.org");
    expect(userEmail).toBeInTheDocument();

    const userCombobox = within(userRoleMember).getByText(/^User$/i);
    await userEvent.click(userCombobox);

    const dropdown = within(userRoleMember).getByTestId(
      "organization-member-role-context-menu",
    );

    // Check that remove option exists
    const removeOption = within(dropdown).getByTestId("remove-option");
    expect(removeOption).toBeInTheDocument();

    await userEvent.click(removeOption);

    expect(removeMemberSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
      userId: "3",
    });

    // Verify the user is no longer in the list
    await waitFor(() => {
      const updatedMemberListItems = screen.getAllByTestId("member-item");
      expect(updatedMemberListItems).toHaveLength(initialMemberCount - 1);
    });

    // Verify the specific user email is no longer present
    expect(screen.queryByText("charlie@acme.org")).not.toBeInTheDocument();
  });

  it.todo(
    "should not allow a user to change another user's role if they are the same role",
  );

  describe("Inviting Organization Members", () => {
    it("should render an invite organization member button", async () => {
      await setupInviteTest();

      const inviteButton = await findInviteButton();
      expect(inviteButton).toBeInTheDocument();
    });

    it("should render a modal when the invite button is clicked", async () => {
      await setupInviteTest();

      expect(screen.queryByTestId("invite-modal")).not.toBeInTheDocument();
      const inviteButton = await findInviteButton();
      await userEvent.click(inviteButton);

      const portalRoot = screen.getByTestId("portal-root");
      expect(
        within(portalRoot).getByTestId("invite-modal"),
      ).toBeInTheDocument();
    });

    it("should close the modal when the close button is clicked", async () => {
      await setupInviteTest();

      const inviteButton = await findInviteButton();
      await userEvent.click(inviteButton);

      const modal = screen.getByTestId("invite-modal");
      const closeButton = within(modal).getByText("BUTTON$CANCEL");
      await userEvent.click(closeButton);

      expect(screen.queryByTestId("invite-modal")).not.toBeInTheDocument();
    });

    it("should render a list item in an invited state when a the user is is invited", async () => {
      const getOrganizationMembersSpy = vi.spyOn(
        organizationService,
        "getOrganizationMembers",
      );

      getOrganizationMembersSpy.mockResolvedValue([
        {
          id: "4",
          email: "tom@acme.org",
          role: "user",
          status: "invited",
        },
      ]);

      await setupInviteTest();

      const members = await screen.findAllByTestId("member-item");
      expect(members).toHaveLength(1);

      const invitedMember = members[0];
      expect(invitedMember).toBeInTheDocument();

      // should have an "invited" badge
      const invitedBadge = within(invitedMember).getByText(/invited/i);
      expect(invitedBadge).toBeInTheDocument();

      // should not have a role combobox
      await userEvent.click(within(invitedMember).getByText(/^User$/i));
      expect(
        within(invitedMember).queryByTestId(
          "organization-member-role-context-menu",
        ),
      ).not.toBeInTheDocument();
    });
  });

  describe("Role-based invite permission behavior", () => {
    it.each([
      { role: "owner" as const, roleName: "Owner" },
      { role: "admin" as const, roleName: "Admin" },
    ])(
      "should show invite button when user has canInviteUsers permission ($roleName role)",
      async ({ role }) => {
        getMeSpy.mockResolvedValue({
          id: "1",
          email: "test@example.com",
          role,
          status: "active",
        });

        await setupTestWithOrg(0);

        const inviteButton = await findInviteButton();

        expect(inviteButton).toBeInTheDocument();
        expect(inviteButton).not.toBeDisabled();
      },
    );

    it("should not show invite button when user lacks canInviteUsers permission (User role)", async () => {
      const userData = {
        id: "1",
        email: "test@example.com",
        role: "user" as const,
        status: "active" as const,
      };

      // Set mock and remove cached query before rendering
      getMeSpy.mockResolvedValue(userData);
      // Remove any cached "me" queries so fresh data is fetched
      queryClient.removeQueries({ queryKey: ["organizations"] });

      await setupTestWithOrg(0);

      // Directly set the query data to force component re-render with user role
      // This ensures the component uses the user role data instead of cached admin data
      queryClient.setQueryData(["organizations", "1", "me"], userData);

      // Wait for the component to update with the new query data
      await waitFor(
        () => {
          const inviteButton = screen.queryByRole("button", {
            name: /ORG\$INVITE_ORGANIZATION_MEMBER/i,
          });
          expect(inviteButton).not.toBeInTheDocument();
        },
        { timeout: 3000 },
      );
    });
  });

  describe("Role-based role change permission behavior", () => {
    it("should not allow an owner to change another owner's role", async () => {
      await verifyRoleChangeNotPermitted(
        {
          id: "1", // Alice is owner in org 1
          email: "alice@acme.org",
          role: "owner",
          status: "active",
        },
        0,
        0, // First member is owner
        "owner",
      );
    });

    it("Owner should see all three role options (owner, admin, user) in dropdown regardless of target member's role", async () => {
      await setupTestWithUserAndOrg(
        {
          id: "1", // Alice is owner in org 1
          email: "alice@acme.org",
          role: "owner",
          status: "active",
        },
        0,
      );

      const memberListItems = await screen.findAllByTestId("member-item");

      // Test with admin member
      const adminMember = memberListItems[1]; // Second member is admin (bob@acme.org)
      const adminDropdown = await openRoleDropdown(adminMember, "admin");

      // Verify all three role options are present for admin member
      expectAllRoleOptionsPresent(adminDropdown);

      // Close dropdown by clicking outside
      await closeDropdown();

      // Test with user member
      const userMember = await findMemberByEmail("charlie@acme.org");
      const userDropdown = await openRoleDropdown(userMember, "user");

      // Verify all three role options are present for user member
      expectAllRoleOptionsPresent(userDropdown);
    });

    it("Admin should not see owner option in role dropdown for any member", async () => {
      await setupTestWithUserAndOrg(
        {
          id: "7", // Ray is admin in org 3
          email: "ray@all-hands.dev",
          role: "admin",
          status: "active",
        },
        2, // org 3
      );

      const memberListItems = await screen.findAllByTestId("member-item");

      // Check user member dropdown
      const userMember = memberListItems[2]; // user member
      const userDropdown = await openRoleDropdown(userMember, "user");
      expectOwnerOptionNotPresent(userDropdown);
      await closeDropdown();

      // Check another user member dropdown if exists
      if (memberListItems.length > 3) {
        const anotherUserMember = memberListItems[3]; // another user member
        const anotherUserDropdown = await openRoleDropdown(
          anotherUserMember,
          "user",
        );
        expectOwnerOptionNotPresent(anotherUserDropdown);
      }
    });

    it("Owner should be able to change any member's role to owner", async () => {
      await setupTestWithUserAndOrg(
        {
          id: "1", // Alice is owner in org 1
          email: "alice@acme.org",
          role: "owner",
          status: "active",
        },
        0,
      );

      const updateMemberRoleSpy = createUpdateMemberRoleSpy();

      const memberListItems = await screen.findAllByTestId("member-item");

      // Test changing admin to owner
      const adminMember = memberListItems[1]; // Second member is admin (bob@acme.org)
      await changeMemberRole(adminMember, "admin", "owner");

      expect(updateMemberRoleSpy).toHaveBeenNthCalledWith(1, {
        userId: "2",
        orgId: "1",
        role: "owner",
      });

      // Test changing user to owner
      const userMember = await findMemberByEmail("charlie@acme.org");
      await changeMemberRole(userMember, "user", "owner");

      expect(updateMemberRoleSpy).toHaveBeenNthCalledWith(2, {
        userId: "3",
        orgId: "1",
        role: "owner",
      });
    });

    it.each([
      {
        description:
          "Owner should be able to change admin's role to admin (no change)",
        userData: {
          id: "1", // Alice is owner in org 1
          email: "alice@acme.org",
          role: "owner" as const,
          status: "active" as const,
        },
        orgIndex: 0,
        memberEmail: "bob@acme.org",
        currentRole: "admin",
        newRole: "admin",
        expectedApiCall: {
          userId: "2",
          orgId: "1",
          role: "admin" as const,
        },
      },
      {
        description:
          "Admin should be able to change user's role to user (no change)",
        userData: {
          id: "7", // Ray is admin in org 3
          email: "ray@all-hands.dev",
          role: "admin" as const,
          status: "active" as const,
        },
        orgIndex: 2, // org 3
        memberEmail: "stephan@all-hands.dev",
        currentRole: "user",
        newRole: "user",
        expectedApiCall: {
          userId: "9",
          orgId: "3",
          role: "user" as const,
        },
      },
      {
        description: "Admin should be able to change user's role to admin",
        userData: {
          id: "7", // Ray is admin in org 3
          email: "ray@all-hands.dev",
          role: "admin" as const,
          status: "active" as const,
        },
        orgIndex: 2, // org 3
        memberEmail: "stephan@all-hands.dev",
        currentRole: "user",
        newRole: "admin",
        expectedApiCall: {
          userId: "9",
          orgId: "3",
          role: "admin" as const,
        },
      },
    ])(
      "$description",
      async ({
        userData,
        orgIndex,
        memberEmail,
        currentRole,
        newRole,
        expectedApiCall,
      }) => {
        await setupTestWithUserAndOrg(userData, orgIndex);

        const updateMemberRoleSpy = createUpdateMemberRoleSpy();

        const member = await findMemberByEmail(memberEmail);

        await changeMemberRole(member, currentRole, newRole);

        expect(updateMemberRoleSpy).toHaveBeenCalledExactlyOnceWith(
          expectedApiCall,
        );
      },
    );
  });
});
