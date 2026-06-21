import { within, screen, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { InviteOrganizationMemberModal } from "#/components/features/org/invite-organization-member-modal";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

vi.mock("react-router", () => ({
  useRevalidator: vi.fn(() => ({ revalidate: vi.fn() })),
}));

const renderInviteOrganizationMemberModal = (config?: {
  onClose: () => void;
}) =>
  render(
    <InviteOrganizationMemberModal onClose={config?.onClose || vi.fn()} />,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      ),
    },
  );

describe("InviteOrganizationMemberModal", () => {
  beforeEach(() => {
    useSelectedOrganizationStore.setState({ organizationId: "1" });
    vi.spyOn(organizationService, "getPendingInvitations").mockResolvedValue({
      items: [],
      email_delivery_configured: false,
      auto_add_enabled: false,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    useSelectedOrganizationStore.setState({ organizationId: null });
  });

  it("should call onClose the modal when the close button is clicked", async () => {
    const onCloseMock = vi.fn();
    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");
    const closeButton = within(modal).getByRole("button", {
      name: /close/i,
    });
    await userEvent.click(closeButton);

    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should call the batch API to invite a single team member when the form is submitted", async () => {
    const inviteMembersBatchSpy = vi.spyOn(
      organizationService,
      "inviteMembers",
    );
    const onCloseMock = vi.fn();

    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");

    const badgeInput = within(modal).getByTestId("emails-badge-input");
    await userEvent.type(badgeInput, "someone@acme.org ");

    // Verify badge is displayed
    expect(screen.getByText("someone@acme.org")).toBeInTheDocument();

    const submitButton = within(modal).getByRole("button", {
      name: /add/i,
    });
    await userEvent.click(submitButton);

    expect(inviteMembersBatchSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
      emails: ["someone@acme.org"],
      role: "member",
    });

    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should invite with the admin role when selected in the role dropdown", async () => {
    const inviteMembersBatchSpy = vi.spyOn(
      organizationService,
      "inviteMembers",
    );
    const onCloseMock = vi.fn();

    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");
    const badgeInput = within(modal).getByTestId("emails-badge-input");
    await userEvent.type(badgeInput, "someone@acme.org ");

    const roleDropdown = within(modal).getByTestId("invite-role-dropdown");
    await userEvent.click(within(roleDropdown).getByTestId("dropdown-trigger"));
    const listbox = await screen.findByRole("listbox");
    await userEvent.click(within(listbox).getByText("ORG$ROLE_ADMIN"));

    const submitButton = within(modal).getByRole("button", { name: /add/i });
    await userEvent.click(submitButton);

    expect(inviteMembersBatchSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
      emails: ["someone@acme.org"],
      role: "admin",
    });
  });

  it("should allow adding multiple emails using badge input and make a batch POST request", async () => {
    const inviteMembersBatchSpy = vi.spyOn(
      organizationService,
      "inviteMembers",
    );
    const onCloseMock = vi.fn();

    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");

    // Should have badge input instead of regular input
    const badgeInput = within(modal).getByTestId("emails-badge-input");
    expect(badgeInput).toBeInTheDocument();

    // Add first email by typing and pressing space
    await userEvent.type(badgeInput, "user1@acme.org ");

    // Add second email by typing and pressing space
    await userEvent.type(badgeInput, "user2@acme.org ");

    // Add third email by typing and pressing space
    await userEvent.type(badgeInput, "user3@acme.org ");

    // Verify badges are displayed
    expect(screen.getByText("user1@acme.org")).toBeInTheDocument();
    expect(screen.getByText("user2@acme.org")).toBeInTheDocument();
    expect(screen.getByText("user3@acme.org")).toBeInTheDocument();

    const submitButton = within(modal).getByRole("button", {
      name: /add/i,
    });
    await userEvent.click(submitButton);

    // Should call batch invite API with all emails
    expect(inviteMembersBatchSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
      emails: ["user1@acme.org", "user2@acme.org", "user3@acme.org"],
      role: "member",
    });

    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should display an error toast when clicking add button with no emails added", async () => {
    // Arrange
    const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");
    const inviteMembersSpy = vi.spyOn(organizationService, "inviteMembers");
    renderInviteOrganizationMemberModal();

    // Act
    const modal = screen.getByTestId("invite-modal");
    const submitButton = within(modal).getByRole("button", { name: /add/i });
    await userEvent.click(submitButton);

    // Assert
    expect(displayErrorToastSpy).toHaveBeenCalledWith(
      "ORG$NO_EMAILS_ADDED_HINT",
    );
    expect(inviteMembersSpy).not.toHaveBeenCalled();
  });

  it("should show invite links instead of closing when email delivery is not configured", async () => {
    vi.spyOn(organizationService, "inviteMembers").mockResolvedValue({
      successful: [
        {
          id: 1,
          email: "someone@acme.org",
          role: "member",
          status: "pending",
          created_at: "2026-01-01T00:00:00Z",
          expires_at: "2026-01-08T00:00:00Z",
          invite_url:
            "https://app.example.com/api/organizations/members/invite/accept?token=inv-abc",
        },
      ],
      failed: [],
      email_delivery_configured: false,
    });
    const onCloseMock = vi.fn();

    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");
    const badgeInput = within(modal).getByTestId("emails-badge-input");
    await userEvent.type(badgeInput, "someone@acme.org ");
    const submitButton = within(modal).getByRole("button", { name: /add/i });
    await userEvent.click(submitButton);

    // The links are the only way the invitee can join, so the modal stays
    // open showing them with copy buttons instead of closing.
    const linksModal = await screen.findByTestId("invite-links-modal");
    expect(onCloseMock).not.toHaveBeenCalled();
    expect(
      within(linksModal).getByTestId("copy-invite-link-button"),
    ).toBeInTheDocument();
    expect(
      within(linksModal).getByText("someone@acme.org"),
    ).toBeInTheDocument();
  });

  it("should show the auto-add hint when sign-in already adds users to the org", async () => {
    vi.spyOn(organizationService, "getPendingInvitations").mockResolvedValue({
      items: [],
      email_delivery_configured: false,
      auto_add_enabled: true,
    });

    renderInviteOrganizationMemberModal();

    expect(
      await screen.findByTestId("auto-add-enabled-hint"),
    ).toBeInTheDocument();
  });

  it("should show failures without the share-links hint when email is configured but some invites fail", async () => {
    vi.spyOn(organizationService, "inviteMembers").mockResolvedValue({
      successful: [
        {
          id: 1,
          email: "ok@acme.org",
          role: "member",
          status: "pending",
          created_at: "2026-01-01T00:00:00Z",
          expires_at: "2026-01-08T00:00:00Z",
          invite_url: "https://app.example.com/accept?token=inv-ok",
        },
      ],
      failed: [{ email: "bad@acme.org", error: "User is already a member" }],
      email_delivery_configured: true,
    });
    const onCloseMock = vi.fn();

    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");
    const badgeInput = within(modal).getByTestId("emails-badge-input");
    await userEvent.type(badgeInput, "ok@acme.org bad@acme.org ");
    await userEvent.click(within(modal).getByRole("button", { name: /add/i }));

    // Modal stays open so the inviter can see what failed...
    const linksModal = await screen.findByTestId("invite-links-modal");
    expect(onCloseMock).not.toHaveBeenCalled();
    expect(within(linksModal).getByText("bad@acme.org")).toBeInTheDocument();
    // ...but the "share these links" hint is wrong when emails were sent.
    expect(
      within(linksModal).queryByText("ORG$INVITATIONS_CREATED_SHARE_LINKS"),
    ).not.toBeInTheDocument();
  });
  it("should invite a typed email even when it was never committed with space", async () => {
    const inviteMembersBatchSpy = vi.spyOn(
      organizationService,
      "inviteMembers",
    );
    const onCloseMock = vi.fn();

    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");
    const badgeInput = within(modal).getByTestId("emails-badge-input");
    // No trailing space — clicking the button blurs the input, which commits
    // the pending text. Previously this errored with "press space".
    await userEvent.type(badgeInput, "someone@acme.org");

    const submitButton = within(modal).getByRole("button", { name: /add/i });
    await userEvent.click(submitButton);

    expect(inviteMembersBatchSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
      emails: ["someone@acme.org"],
      role: "member",
    });
  });
});
