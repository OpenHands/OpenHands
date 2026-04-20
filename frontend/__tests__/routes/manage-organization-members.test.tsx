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
  MOCK_TEAM_ORG_ACME,
  INITIAL_MOCK_ORGS,
} from "#/mocks/org-handlers";
import OptionService from "#/api/option-service/option-service.api";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { createMockWebClientConfig } from "#/mocks/settings-handlers";

const mockQueryClient = vi.hoisted(() => {
  const { QueryClient } = require("@tanstack/react-query");
  return new QueryClient();
});

vi.mock("#/query-client-config", () => ({
  queryClient: mockQueryClient,
}));

vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          ORG$SELECT_ORGANIZATION_PLACEHOLDER: "Please select an organization",
          ORG$PERSONAL_WORKSPACE: "Personal Workspace",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: vi.fn(),
      },
    }),
  };
});

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

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
        handle: { hideTitle: true },
      },
      {
        Component: () => <div data-testid="user-settings" />,
        path: "/settings/member",
      },
    ],
  },
]);

let queryClient: QueryClient;

describe("Manage Organization Members Route", () => {
  const getMeSpy = vi.spyOn(organizationService, "getMe");

  beforeEach(() => {
    // Set Zustand store to a team org so clientLoader allows access to /settings/org-members
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    // Seed organizations into the module-level queryClient used by clientLoader
    mockQueryClient.setQueryData(["organizations"], {
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });

    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(
      createMockWebClientConfig({
        app_mode: "saas",
        feature_flags: {
          enable_billing: true,
          hide_llm_settings: false,
          enable_jira: false,
          enable_jira_dc: false,
          enable_linear: false,
          hide_users_page: false,
          hide_billing_page: false,
          hide_integrations_page: false,
        },
      }),
    );

    queryClient = new QueryClient();

    // Pre-seed organizations so org selector renders immediately (avoids flaky race with API fetch)
    queryClient.setQueryData(["organizations"], {
      items: INITIAL_MOCK_ORGS,
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });

    // Set default mock for user (admin role has invite permission)
    getMeSpy.mockResolvedValue({
      org_id: "1",
      user_id: "1",
      email: "test@example.com",
      role: "admin",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_base_url: "https://api.openai.com",
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
    // Reset Zustand store and module-level queryClient
    useSelectedOrganizationStore.setState({ organizationId: null });
    mockQueryClient.clear();
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

    // If role is changing, confirm the modal
    if (currentRole.toLowerCase() !== newRole.toLowerCase()) {
      const confirmButton = await screen.findByTestId("confirm-button");
      await userEvent.click(confirmButton);
    }
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
      org_id: string;
      user_id: string;
      email: string;
      role: "owner" | "admin" | "member";
      llm_api_key: string;
      max_iterations: number;
      llm_model: string;
      llm_base_url: string;
      status: "active" | "invited" | "inactive";
    },
    orgIndex: number,
  ) => {
    getMeSpy.mockResolvedValue(userData);
    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");
    await selectOrganization({ orgIndex });
    // Wait for member list to be rendered (async data loaded)
    await screen.findAllByTestId("member-item");
  };

  // Helper function to create updateMember spy
  const createUpdateMemberRoleSpy = () =>
    vi.spyOn(organizationService, "updateMember");

  // Helper function to verify role change is not permitted
  const verifyRoleChangeNotPermitted = async (
    userData: {
      org_id: string;
      user_id: string;
      email: string;
      role: "owner" | "admin" | "member";
      llm_api_key: string;
      max_iterations: number;
      llm_model: string;
      llm_base_url: string;
      status: "active" | "invited" | "inactive";
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
    await screen.findByTestId("manage-organization-members-settings");
    await selectOrganization({ orgIndex });
  };

  // Helper function to setup test with organization (waits for settings screen)
  const setupTestWithOrg = async (orgIndex: number = 0) => {
    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");
    await selectOrganization({ orgIndex });
  };

  // Helper function to find invite button
  const findInviteButton = () =>
    screen.findByRole("button", {
      name: /ORG\$INVITE_ORG_MEMBERS/i,
    });

  // Helper function to verify all three role options are present in dropdown
  const expectAllRoleOptionsPresent = (dropdown: HTMLElement) => {
    expect(within(dropdown).getByText(/owner/i)).toBeInTheDocument();
    expect(within(dropdown).getByText(/admin/i)).toBeInTheDocument();
    expect(within(dropdown).getByText(/member/i)).toBeInTheDocument();
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
    // @ts-expect-error - partial mock for testing
    getConfigSpy.mockResolvedValue({
      app_mode: "oss",
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

    // First org is auto-selected, so members are fetched for org "1"
    await selectOrganization({ orgIndex: 1 }); // Acme Corp
    expect(getOrganizationMembersSpy).toHaveBeenLastCalledWith({
      orgId: "2",
      page: 1,
      limit: 10,
      email: undefined,
    });
  });

  it("should render the list of organization members", async () => {
    await setupTestWithOrg(0);
    const members = ORGS_AND_MEMBERS["1"];

    // Wait for org "1" member to appear (ensures org switch is complete)
    // This is needed because placeholderData: keepPreviousData shows stale data during transitions
    await screen.findByText(members[0].email);

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
        org_id: "1",
        user_id: "1",
        email: "test@example.com",
        role: "admin",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active",
      },
      1, // Acme Corp (org "2") - has owner, admin, user
    );

    const updateMemberRoleSpy = createUpdateMemberRoleSpy();

    const memberListItems = await screen.findAllByTestId("member-item");
    const userRoleMember = memberListItems[2]; // third member is "user" (charlie)

    let userCombobox = within(userRoleMember).getByText(/^Member$/i);
    expect(userCombobox).toBeInTheDocument();

    // Change role from user to admin
    await changeMemberRole(userRoleMember, "member", "admin");

    expect(updateMemberRoleSpy).toHaveBeenCalledExactlyOnceWith({
      userId: "3", // charlie's id
      orgId: "2",
      role: "admin",
    });
    expectDropdownNotVisible(userRoleMember);

    // Verify the role has been updated in the UI
    userCombobox = within(userRoleMember).getByText(/^Admin$/i);
    expect(userCombobox).toBeInTheDocument();

    // Revert the role back to user
    await changeMemberRole(userRoleMember, "admin", "member");

    expect(updateMemberRoleSpy).toHaveBeenNthCalledWith(2, {
      userId: "3",
      orgId: "2",
      role: "member",
    });

    // Verify the role has been reverted in the UI
    userCombobox = within(userRoleMember).getByText(/^Member$/i);
    expect(userCombobox).toBeInTheDocument();
  });

  it("should not allow an admin to change the owner's role", async () => {
    // User is bob (admin, user_id: "2") trying to edit alice (owner, user_id: "1")
    // Admins don't have change_user_role:owner permission, so dropdown shouldn't show

    // Reset mock data to ensure clean state
    resetOrgsAndMembersMockData();

    // Pre-seed the /me query data to avoid stale cache issues
    const userData = {
      org_id: "2",
      user_id: "2", // bob (admin) - different from alice
      email: "bob@acme.org",
      role: "admin" as const,
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_base_url: "https://api.openai.com",
      status: "active" as const,
    };

    getMeSpy.mockResolvedValue(userData);
    queryClient.setQueryData(["organizations", "2", "me"], userData);

    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");
    await selectOrganization({ orgIndex: 1 }); // Acme Corp (org "2")

    // Wait for member list to load
    const memberListItems = await screen.findAllByTestId("member-item");

    // First member is alice (owner)
    const targetMember = memberListItems[0];
    const roleText = within(targetMember).getByText(/^owner$/i);
    expect(roleText).toBeInTheDocument();
    await userEvent.click(roleText);

    // Verify that the dropdown does not open (admin can't edit owner)
    expectDropdownNotVisible(targetMember);
  });

  it("should allow an admin to change another admin's role", async () => {
    // Mock members to include two admins so we can test admin editing another admin
    const getOrganizationMembersSpy = vi.spyOn(
      organizationService,
      "getOrganizationMembers",
    );
    const getOrganizationMembersCountSpy = vi.spyOn(
      organizationService,
      "getOrganizationMembersCount",
    );

    const twoAdminsMembers = [
      {
        org_id: "2",
        user_id: "1",
        email: "admin1@acme.org",
        role: "admin" as const,
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active" as const,
      },
      {
        org_id: "2",
        user_id: "2",
        email: "admin2@acme.org",
        role: "admin" as const,
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active" as const,
      },
    ];

    getOrganizationMembersSpy.mockResolvedValue({
      items: twoAdminsMembers,
      current_page: 1,
      per_page: 10,
    });
    getOrganizationMembersCountSpy.mockResolvedValue(2);

    // Current user is admin1 (user_id: "1")
    getMeSpy.mockResolvedValue({
      org_id: "2",
      user_id: "1",
      email: "admin1@acme.org",
      role: "admin",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_base_url: "https://api.openai.com",
      status: "active",
    });

    const updateMemberRoleSpy = createUpdateMemberRoleSpy();

    renderManageOrganizationMembers();
    await screen.findByTestId("manage-organization-members-settings");
    await selectOrganization({ orgIndex: 1 }); // Acme Corp

    // Find admin2 and change their role to member
    const admin2Member = await findMemberByEmail("admin2@acme.org");
    await changeMemberRole(admin2Member, "admin", "member");

    expect(updateMemberRoleSpy).toHaveBeenCalledExactlyOnceWith({
      userId: "2",
      orgId: "2",
      role: "member",
    });

    // Restore spies to prevent interference with subsequent tests
    getOrganizationMembersSpy.mockRestore();
    getOrganizationMembersCountSpy.mockRestore();
  });

  it("should not allow a user to change their own role", async () => {
    // Mock the /me endpoint to return a user ID that matches one of the members
    await verifyRoleChangeNotPermitted(
      {
        org_id: "1",
        user_id: "1", // Same as first member from org 1
        email: "alice@acme.org",
        role: "owner",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active",
      },
      0,
      0, // First member (user_id: "1")
      "Owner",
    );
  });

  it("should show a remove option in the role dropdown and remove the user from the list", async () => {
    const removeMemberSpy = vi.spyOn(organizationService, "removeMember");

    await setupTestWithOrg(1); // Acme Corp (org "2") - has owner, admin, user

    // Get initial member count
    const memberListItems = await screen.findAllByTestId("member-item");
    const initialMemberCount = memberListItems.length;

    const userRoleMember = memberListItems[2]; // third member is "user"
    const userEmail = within(userRoleMember).getByText("charlie@acme.org");
    expect(userEmail).toBeInTheDocument();

    const userCombobox = within(userRoleMember).getByText(/^Member$/i);
    await userEvent.click(userCombobox);

    const dropdown = within(userRoleMember).getByTestId(
      "organization-member-role-context-menu",
    );

    // Check that remove option exists
    const removeOption = within(dropdown).getByTestId("remove-option");
    expect(removeOption).toBeInTheDocument();

    await userEvent.click(removeOption);

    // Wait for confirmation modal to appear and click confirm
    const confirmButton = await screen.findByTestId("confirm-button");
    await userEvent.click(confirmButton);

    expect(removeMemberSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "2",
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
      const closeButton = within(modal).getByText("BUTTON$CLOSE");
      await userEvent.click(closeButton);

      expect(screen.queryByTestId("invite-modal")).not.toBeInTheDocument();
    });

    it("should render a list item in an invited state when a the user is is invited", async () => {
      const getOrganizationMembersSpy = vi.spyOn(
        organizationService,
        "getOrganizationMembers",
      );
      const getOrganizationMembersCountSpy = vi.spyOn(
        organizationService,
        "getOrganizationMembersCount",
      );

      getOrganizationMembersSpy.mockResolvedValue({
        items: [
          {
            org_id: "1",
            user_id: "4",
            email: "tom@acme.org",
            role: "member",
            llm_api_key: "**********",
            max_iterations: 20,
            llm_model: "gpt-4",
            llm_base_url: "https://api.openai.com",
            status: "invited",
          },
        ],
        current_page: 1,
        per_page: 10,
      });
      getOrganizationMembersCountSpy.mockResolvedValue(1);

      await setupInviteTest();

      const members = await screen.findAllByTestId("member-item");
      expect(members).toHaveLength(1);

      const invitedMember = members[0];
      expect(invitedMember).toBeInTheDocument();

      // should have an "invited" badge
      const invitedBadge = within(invitedMember).getByText(/invited/i);
      expect(invitedBadge).toBeInTheDocument();

      // should not have a role combobox
      await userEvent.click(within(invitedMember).getByText(/^Member$/i));
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
          org_id: "1",
          user_id: "1",
          email: "test@example.com",
          role,
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
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
        org_id: "1",
        user_id: "1",
        email: "test@example.com",
        role: "member" as const,
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
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
            name: /ORG\$INVITE_ORG_MEMBERS/i,
          });
          expect(inviteButton).not.toBeInTheDocument();
        },
        { timeout: 3000 },
      );
    });
  });

  describe("Role-based role change permission behavior", () => {
    it("should not allow an owner to change their own role", async () => {
      // Acme Corp (org "2") - alice is owner, can't change her own role

      // Reset mock data to ensure clean state
      resetOrgsAndMembersMockData();

      // Pre-seed the /me query data to avoid stale cache issues
      const userData = {
        org_id: "2",
        user_id: "1", // alice (owner) - same as first member
        email: "alice@acme.org",
        role: "owner" as const,
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active" as const,
      };

      getMeSpy.mockResolvedValue(userData);
      queryClient.setQueryData(["organizations", "2", "me"], userData);

      renderManageOrganizationMembers();
      await screen.findByTestId("manage-organization-members-settings");
      await selectOrganization({ orgIndex: 1 }); // Acme Corp (org "2")

      // Wait for member list to load
      const memberListItems = await screen.findAllByTestId("member-item");

      // First member is alice (owner) - same as current user
      const targetMember = memberListItems[0];
      const roleText = within(targetMember).getByText(/^owner$/i);
      expect(roleText).toBeInTheDocument();
      await userEvent.click(roleText);

      // Verify that the dropdown does not open (can't edit own role)
      expectDropdownNotVisible(targetMember);
    });

    it("should allow an owner to change another owner's role", async () => {
      // Mock members to include two owners so we can test owner editing another owner
      const getOrganizationMembersSpy = vi.spyOn(
        organizationService,
        "getOrganizationMembers",
      );
      const getOrganizationMembersCountSpy = vi.spyOn(
        organizationService,
        "getOrganizationMembersCount",
      );

      const twoOwnersMembers = [
        {
          org_id: "2",
          user_id: "1",
          email: "owner1@acme.org",
          role: "owner" as const,
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
          status: "active" as const,
        },
        {
          org_id: "2",
          user_id: "2",
          email: "owner2@acme.org",
          role: "owner" as const,
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
          status: "active" as const,
        },
      ];

      getOrganizationMembersSpy.mockResolvedValue({
        items: twoOwnersMembers,
        current_page: 1,
        per_page: 10,
      });
      getOrganizationMembersCountSpy.mockResolvedValue(2);

      // Current user is owner1 (user_id: "1")
      getMeSpy.mockResolvedValue({
        org_id: "2",
        user_id: "1",
        email: "owner1@acme.org",
        role: "owner",
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_base_url: "https://api.openai.com",
        status: "active",
      });

      const updateMemberRoleSpy = createUpdateMemberRoleSpy();

      renderManageOrganizationMembers();
      await screen.findByTestId("manage-organization-members-settings");
      await selectOrganization({ orgIndex: 1 }); // Acme Corp

      // Find owner2 and change their role to admin
      const owner2Member = await findMemberByEmail("owner2@acme.org");
      await changeMemberRole(owner2Member, "owner", "admin");

      expect(updateMemberRoleSpy).toHaveBeenCalledExactlyOnceWith({
        userId: "2",
        orgId: "2",
        role: "admin",
      });

      // Restore spies to prevent interference with subsequent tests
      getOrganizationMembersSpy.mockRestore();
      getOrganizationMembersCountSpy.mockRestore();
    });

    it("Owner should see all three role options (owner, admin, user) in dropdown regardless of target member's role", async () => {
      await setupTestWithUserAndOrg(
        {
          org_id: "1",
          user_id: "1", // First member is owner in org 1
          email: "alice@acme.org",
          role: "owner",
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
          status: "active",
        },
        1, // Acme Corp (org "2")
      );

      const memberListItems = await screen.findAllByTestId("member-item");

      // Test with admin member
      const adminMember = memberListItems[1]; // Second member is admin (user_id: "2")
      const adminDropdown = await openRoleDropdown(adminMember, "admin");

      // Verify all three role options are present for admin member
      expectAllRoleOptionsPresent(adminDropdown);

      // Close dropdown by clicking outside
      await closeDropdown();

      // Test with user member
      const userMember = await findMemberByEmail("charlie@acme.org");
      const userDropdown = await openRoleDropdown(userMember, "member");

      // Verify all three role options are present for user member
      expectAllRoleOptionsPresent(userDropdown);
    });

    it("Admin should not see owner option in role dropdown for any member", async () => {
      await setupTestWithUserAndOrg(
        {
          org_id: "3",
          user_id: "7", // Ray is admin in org 3
          email: "ray@all-hands.dev",
          role: "admin",
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
          status: "active",
        },
        3, // All Hands AI (org "4")
      );

      const memberListItems = await screen.findAllByTestId("member-item");

      // Check user member dropdown
      const userMember = memberListItems[2]; // user member
      const userDropdown = await openRoleDropdown(userMember, "member");
      expectOwnerOptionNotPresent(userDropdown);
      await closeDropdown();

      // Check another user member dropdown (stephan is at index 3)
      if (memberListItems.length > 3) {
        const anotherUserMember = memberListItems[3]; // stephan@all-hands.dev
        const anotherUserDropdown = await openRoleDropdown(
          anotherUserMember,
          "member",
        );
        expectOwnerOptionNotPresent(anotherUserDropdown);
      }
    });

    it("Owner should be able to change any member's role to owner", async () => {
      await setupTestWithUserAndOrg(
        {
          org_id: "1",
          user_id: "1", // First member is owner in org 1
          email: "alice@acme.org",
          role: "owner",
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
          status: "active",
        },
        1, // Acme Corp (org "2")
      );

      const updateMemberRoleSpy = createUpdateMemberRoleSpy();

      const memberListItems = await screen.findAllByTestId("member-item");

      // Test changing admin to owner
      const adminMember = memberListItems[1]; // Second member is admin (user_id: "2")
      await changeMemberRole(adminMember, "admin", "owner");

      expect(updateMemberRoleSpy).toHaveBeenNthCalledWith(1, {
        userId: "2",
        orgId: "2",
        role: "owner",
      });

      // Test changing user to owner
      const userMember = await findMemberByEmail("charlie@acme.org");
      await changeMemberRole(userMember, "member", "owner");

      expect(updateMemberRoleSpy).toHaveBeenNthCalledWith(2, {
        userId: "3",
        orgId: "2",
        role: "owner",
      });
    });

    it("Admin should be able to change member's role to admin", async () => {
      await setupTestWithUserAndOrg(
        {
          org_id: "4",
          user_id: "7", // Ray is admin in org 4
          email: "ray@all-hands.dev",
          role: "admin" as const,
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
          status: "active" as const,
        },
        3, // All Hands AI (org "4")
      );

      const updateMemberRoleSpy = createUpdateMemberRoleSpy();

      const member = await findMemberByEmail("stephan@all-hands.dev");

      await changeMemberRole(member, "member", "admin");

      expect(updateMemberRoleSpy).toHaveBeenCalledExactlyOnceWith({
        userId: "9",
        orgId: "4",
        role: "admin" as const,
      });
    });

    it("should not show confirmation modal or call API when selecting the same role", async () => {
      await setupTestWithUserAndOrg(
        {
          org_id: "1",
          user_id: "1", // First member is owner in org 1
          email: "alice@acme.org",
          role: "owner" as const,
          llm_api_key: "**********",
          max_iterations: 20,
          llm_model: "gpt-4",
          llm_base_url: "https://api.openai.com",
          status: "active" as const,
        },
        1, // Acme Corp (org "2")
      );

      const updateMemberRoleSpy = createUpdateMemberRoleSpy();

      const member = await findMemberByEmail("bob@acme.org");

      // Open dropdown and select the same role (admin -> admin)
      const dropdown = await openRoleDropdown(member, "admin");
      const roleOption = within(dropdown).getByText(/admin/i);
      await userEvent.click(roleOption);

      // Verify no confirmation modal appears
      expect(screen.queryByTestId("confirm-button")).not.toBeInTheDocument();

      // Verify no API call was made
      expect(updateMemberRoleSpy).not.toHaveBeenCalled();
    });
  });
});
