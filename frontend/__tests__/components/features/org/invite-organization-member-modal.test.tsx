import { within, screen, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { InviteOrganizationMemberModal } from "#/components/features/org/invite-organization-member-modal";

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

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: vi.fn(() => ({
    orgId: "1",
    setOrgId: vi.fn(),
  })),
}));

describe("InviteOrganizationMemberModal", () => {
  it("should call onClose the modal when the close button is clicked", async () => {
    const onCloseMock = vi.fn();
    renderInviteOrganizationMemberModal({ onClose: onCloseMock });

    const modal = screen.getByTestId("invite-modal");
    const closeButton = within(modal).getByRole("button", {
      name: /skip/i,
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
      name: /next/i,
    });
    await userEvent.click(submitButton);

    expect(inviteMembersBatchSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
      emails: ["someone@acme.org"],
    });

    expect(onCloseMock).toHaveBeenCalledOnce();
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
      name: /next/i,
    });
    await userEvent.click(submitButton);

    // Should call batch invite API with all emails
    expect(inviteMembersBatchSpy).toHaveBeenCalledExactlyOnceWith({
      orgId: "1",
      emails: ["user1@acme.org", "user2@acme.org", "user3@acme.org"],
    });

    expect(onCloseMock).toHaveBeenCalledOnce();
  });
});
