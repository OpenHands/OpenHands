import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, test, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { UserContextMenu } from "#/components/features/user/user-context-menu";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { GetComponentPropTypes } from "#/utils/get-component-prop-types";
import { INITIAL_MOCK_ORGS } from "#/mocks/org-handlers";
import AuthService from "#/api/auth-service/auth-service.api";

type UserContextMenuProps = GetComponentPropTypes<typeof UserContextMenu>;

function UserContextMenuWithRootOutlet({
  type,
  onClose,
}: UserContextMenuProps) {
  return (
    <div>
      <div data-testid="portal-root" id="portal-root" />
      <UserContextMenu type={type} onClose={onClose} />
    </div>
  );
}

const renderUserContextMenu = ({ type, onClose }: UserContextMenuProps) =>
  render(<UserContextMenuWithRootOutlet type={type} onClose={onClose} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });

const { navigateMock } = vi.hoisted(() => ({
  navigateMock: vi.fn(),
}));

vi.mock("react-router", async (importActual) => ({
  ...(await importActual()),
  useNavigate: () => navigateMock,
  useRevalidator: () => ({
    revalidate: vi.fn(),
  }),
}));

describe("UserContextMenu", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the default context items for a user", () => {
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    screen.getByTestId("org-selector");
    screen.getByText("Logout");
    screen.getByText("Settings");

    expect(screen.queryByText("Invite Team")).not.toBeInTheDocument();
    expect(screen.queryByText("Manage Team")).not.toBeInTheDocument();
    expect(screen.queryByText("Manage Account")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Create New Organization"),
    ).not.toBeInTheDocument();
  });

  it("should render additional context items when user is an admin", () => {
    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    screen.getByTestId("org-selector");
    screen.getByText("Invite Team");
    screen.getByText("Manage Team");
    screen.getByText("Manage Account");

    expect(
      screen.queryByText("Create New Organization"),
    ).not.toBeInTheDocument();
  });

  it("should render additional context items when user is a super admin", () => {
    renderUserContextMenu({ type: "superadmin", onClose: vi.fn });

    screen.getByTestId("org-selector");
    screen.getByText("Invite Team");
    screen.getByText("Manage Team");
    screen.getByText("Manage Account");
    screen.getByText("Create New Organization");
  });

  it("should call the logout handler when Logout is clicked", async () => {
    const logoutSpy = vi.spyOn(AuthService, "logout");
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    const logoutButton = screen.getByText("Logout");
    await userEvent.click(logoutButton);

    expect(logoutSpy).toHaveBeenCalledOnce();
  });

  it("should navigate to /settings when Settings is clicked", async () => {
    renderUserContextMenu({ type: "user", onClose: vi.fn });

    const settingsButton = screen.getByText("Settings");
    await userEvent.click(settingsButton);

    expect(navigateMock).toHaveBeenCalledExactlyOnceWith("/settings");
  });

  it("should navigate to /settings/team when Manage Team is clicked", async () => {
    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    const manageTeamButton = screen.getByText("Manage Team");
    await userEvent.click(manageTeamButton);

    expect(navigateMock).toHaveBeenCalledExactlyOnceWith("/settings/team");
  });

  it("should navigate to /settings/org when Manage Account is clicked", async () => {
    renderUserContextMenu({ type: "admin", onClose: vi.fn });

    const manageTeamButton = screen.getByText("Manage Account");
    await userEvent.click(manageTeamButton);

    expect(navigateMock).toHaveBeenCalledExactlyOnceWith("/settings/org");
  });

  describe("Create New Organization", () => {
    it("should render a modal when Create New Organization is clicked", async () => {
      renderUserContextMenu({ type: "superadmin", onClose: vi.fn });

      expect(screen.queryByTestId("create-org-modal")).not.toBeInTheDocument();

      const createOrgButton = screen.getByText("Create New Organization");
      await userEvent.click(createOrgButton);

      const rootOutlet = screen.getByTestId("portal-root");
      expect(
        within(rootOutlet).getByTestId("create-org-modal"),
      ).toBeInTheDocument();
    });

    it("should close the modal when the close button is clicked", async () => {
      renderUserContextMenu({ type: "superadmin", onClose: vi.fn });

      const createOrgButton = screen.getByText("Create New Organization");
      await userEvent.click(createOrgButton);

      expect(screen.getByTestId("create-org-modal")).toBeInTheDocument();

      // Simulate closing the modal
      const skipButton = screen.getByRole("button", { name: /skip/i });
      await userEvent.click(skipButton);

      expect(screen.queryByTestId("create-org-modal")).not.toBeInTheDocument();
    });

    it("should call the API to create a new organization when the form is submitted", async () => {
      const createOrgSpy = vi.spyOn(organizationService, "createOrganization");
      renderUserContextMenu({ type: "superadmin", onClose: vi.fn });

      const createOrgButton = screen.getByText("Create New Organization");
      await userEvent.click(createOrgButton);

      expect(screen.getByTestId("create-org-modal")).toBeInTheDocument();

      const orgNameInput = screen.getByTestId("org-name-input");
      await userEvent.type(orgNameInput, "New Organization");

      const nextButton = screen.getByRole("button", { name: /next/i });
      await userEvent.click(nextButton);

      expect(createOrgSpy).toHaveBeenCalledExactlyOnceWith({
        name: "New Organization",
      });
      expect(screen.queryByTestId("create-org-modal")).not.toBeInTheDocument();
    });

    it("should automatically select the newly created organization", async () => {
      const createOrgSpy = vi.spyOn(organizationService, "createOrganization");
      renderUserContextMenu({ type: "superadmin", onClose: vi.fn });

      const createOrgButton = screen.getByText("Create New Organization");
      await userEvent.click(createOrgButton);

      const orgNameInput = screen.getByTestId("org-name-input");
      await userEvent.type(orgNameInput, "New Organization");

      const nextButton = screen.getByRole("button", { name: /next/i });
      await userEvent.click(nextButton);

      expect(createOrgSpy).toHaveBeenCalledExactlyOnceWith({
        name: "New Organization",
      });

      // Verify the organization selector now shows the newly created organization
      const orgSelector = screen.getByTestId("org-selector");
      expect(orgSelector.getAttribute("value")).toBe("New Organization");
    });

    it("should show invite modal immediately after creating an organization", async () => {
      const createOrgSpy = vi.spyOn(organizationService, "createOrganization");
      renderUserContextMenu({ type: "superadmin", onClose: vi.fn });

      // Verify invite modal is not visible initially
      expect(screen.queryByTestId("invite-modal")).not.toBeInTheDocument();

      const createOrgButton = screen.getByText("Create New Organization");
      await userEvent.click(createOrgButton);

      const orgNameInput = screen.getByTestId("org-name-input");
      await userEvent.type(orgNameInput, "New Organization");

      const nextButton = screen.getByRole("button", { name: /next/i });
      await userEvent.click(nextButton);

      expect(createOrgSpy).toHaveBeenCalledExactlyOnceWith({
        name: "New Organization",
      });

      // Verify the create org modal is closed
      expect(screen.queryByTestId("create-org-modal")).not.toBeInTheDocument();

      // Verify the invite modal appears immediately
      const portalRoot = screen.getByTestId("portal-root");
      expect(
        within(portalRoot).getByTestId("invite-modal"),
      ).toBeInTheDocument();
    });
  });

  it("should call the onClose handler when clicking outside the context menu", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "user", onClose: onCloseMock });

    const contextMenu = screen.getByTestId("user-context-menu");
    await userEvent.click(contextMenu);

    expect(onCloseMock).not.toHaveBeenCalled();

    // Simulate clicking outside the context menu
    await userEvent.click(document.body);

    expect(onCloseMock).toHaveBeenCalled();
  });

  it("should call the onClose handler after each action", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "superadmin", onClose: onCloseMock });

    const logoutButton = screen.getByText("Logout");
    await userEvent.click(logoutButton);
    expect(onCloseMock).toHaveBeenCalledTimes(1);

    const settingsButton = screen.getByText("Settings");
    await userEvent.click(settingsButton);
    expect(onCloseMock).toHaveBeenCalledTimes(2);

    const manageTeamButton = screen.getByText("Manage Team");
    await userEvent.click(manageTeamButton);
    expect(onCloseMock).toHaveBeenCalledTimes(3);

    const manageAccountButton = screen.getByText("Manage Account");
    await userEvent.click(manageAccountButton);
    expect(onCloseMock).toHaveBeenCalledTimes(4);
  });

  it("should render the invite user modal when Invite Team is clicked", async () => {
    const inviteMembersBatchSpy = vi.spyOn(
      organizationService,
      "inviteMembers",
    );
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "admin", onClose: onCloseMock });

    const inviteButton = screen.getByText("Invite Team");
    await userEvent.click(inviteButton);

    const portalRoot = screen.getByTestId("portal-root");
    expect(within(portalRoot).getByTestId("invite-modal")).toBeInTheDocument();

    await userEvent.click(
      within(portalRoot).getByRole("button", { name: /skip/i }),
    );
    expect(inviteMembersBatchSpy).not.toHaveBeenCalled();
  });

  test("the user can change orgs", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "user", onClose: onCloseMock });

    const orgSelector = screen.getByTestId("org-selector");
    expect(orgSelector).toBeInTheDocument();

    // Simulate changing the organization
    await userEvent.click(orgSelector);
    const orgOption = screen.getByText(INITIAL_MOCK_ORGS[1].name);
    await userEvent.click(orgOption);

    expect(onCloseMock).not.toHaveBeenCalled();

    // Verify that the dropdown shows the selected organization
    // The dropdown should now display the selected org name
    expect(orgSelector).toHaveValue(INITIAL_MOCK_ORGS[1].name);
  });

  it("should have Personal Account as the default selected option with null value", async () => {
    const onCloseMock = vi.fn();
    renderUserContextMenu({ type: "user", onClose: onCloseMock });

    const orgSelector = screen.getByTestId("org-selector");

    // Should default to "Personal Account" when orgId is null
    expect(orgSelector).toHaveValue("Personal Account");

    // Click to open dropdown
    await userEvent.click(orgSelector);

    // Should have "Personal Account" as an option
    const personalAccountOption = screen.getByText("Personal Account");
    expect(personalAccountOption).toBeInTheDocument();

    // Select an organization
    const orgOption = screen.getByText(INITIAL_MOCK_ORGS[1].name);
    await userEvent.click(orgOption);

    // Should now show the selected organization
    expect(orgSelector).toHaveValue(INITIAL_MOCK_ORGS[1].name);

    // Click to open dropdown again
    await userEvent.click(orgSelector);

    // Click on Personal Account to go back
    const personalAccountOptionAgain = screen.getByText("Personal Account");
    await userEvent.click(personalAccountOptionAgain);

    // Should show "Personal Account" after going back
    expect(orgSelector).toHaveValue("Personal Account");
  });
});
