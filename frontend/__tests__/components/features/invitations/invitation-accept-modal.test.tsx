import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, createAxiosError } from "test-utils";
import { InvitationAcceptModal } from "#/components/features/invitations/invitation-accept-modal";
import { organizationService } from "#/api/organization-service/organization-service.api";
import * as toastHandlers from "#/utils/custom-toast-handlers";

// Mock the organization service
vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    acceptInvitation: vi.fn(),
  },
}));

// Mock toast handlers
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

describe("InvitationAcceptModal", () => {
  const mockToken = "test-invitation-token-123";
  const mockOnClose = vi.fn();
  const mockOnSuccess = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the modal with title and description", () => {
    renderWithProviders(
      <InvitationAcceptModal
        token={mockToken}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    expect(screen.getByTestId("invitation-accept-modal")).toBeInTheDocument();
    expect(
      screen.getByText("ORG$INVITATION_ACCEPT_TITLE"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("ORG$INVITATION_ACCEPT_DESCRIPTION"),
    ).toBeInTheDocument();
  });

  it("should render accept and cancel buttons", () => {
    renderWithProviders(
      <InvitationAcceptModal
        token={mockToken}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    expect(screen.getByTestId("accept-invitation-button")).toBeInTheDocument();
    expect(screen.getByTestId("cancel-invitation-button")).toBeInTheDocument();
  });

  it("should call onClose when cancel button is clicked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <InvitationAcceptModal
        token={mockToken}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    await user.click(screen.getByTestId("cancel-invitation-button"));

    expect(mockOnClose).toHaveBeenCalledOnce();
  });

  it("should call acceptInvitation when accept button is clicked", async () => {
    const user = userEvent.setup();
    const mockResponse = {
      success: true,
      org_id: "org-123",
      org_name: "Test Organization",
      role: "member",
    };

    vi.mocked(organizationService.acceptInvitation).mockResolvedValueOnce(
      mockResponse,
    );

    renderWithProviders(
      <InvitationAcceptModal
        token={mockToken}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    await user.click(screen.getByTestId("accept-invitation-button"));

    await waitFor(() => {
      expect(organizationService.acceptInvitation).toHaveBeenCalledWith({
        token: mockToken,
      });
    });
  });

  it("should call onSuccess with org_id and show success toast on successful acceptance", async () => {
    const user = userEvent.setup();
    const mockResponse = {
      success: true,
      org_id: "org-123",
      org_name: "Test Organization",
      role: "member",
    };

    vi.mocked(organizationService.acceptInvitation).mockResolvedValueOnce(
      mockResponse,
    );

    renderWithProviders(
      <InvitationAcceptModal
        token={mockToken}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    await user.click(screen.getByTestId("accept-invitation-button"));

    await waitFor(() => {
      expect(mockOnSuccess).toHaveBeenCalledWith({
        orgId: "org-123",
        orgName: "Test Organization",
        isPersonal: false,
      });
    });

    expect(toastHandlers.displaySuccessToast).toHaveBeenCalled();
  });

  it("should show loading spinner and disable buttons while accepting", async () => {
    const user = userEvent.setup();

    // Create a promise that we can control
    let resolvePromise: (value: unknown) => void;
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    vi.mocked(organizationService.acceptInvitation).mockReturnValueOnce(
      pendingPromise as Promise<{
        success: boolean;
        org_id: string;
        org_name: string;
        role: string;
      }>,
    );

    renderWithProviders(
      <InvitationAcceptModal
        token={mockToken}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    // Click accept to trigger loading state
    await user.click(screen.getByTestId("accept-invitation-button"));

    // Check loading state
    await waitFor(() => {
      expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
    });
    expect(screen.getByTestId("accept-invitation-button")).toBeDisabled();
    expect(screen.getByTestId("cancel-invitation-button")).toBeDisabled();

    // Resolve the promise to clean up
    resolvePromise!({
      success: true,
      org_id: "org-123",
      org_name: "Test Organization",
      role: "member",
    });
  });

  describe("error handling", () => {
    it("should show expired error toast and call onClose when invitation is expired", async () => {
      const user = userEvent.setup();

      vi.mocked(organizationService.acceptInvitation).mockRejectedValueOnce(
        createAxiosError(400, "Bad Request", { detail: "invitation_expired" }),
      );

      renderWithProviders(
        <InvitationAcceptModal
          token={mockToken}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />,
      );

      await user.click(screen.getByTestId("accept-invitation-button"));

      await waitFor(() => {
        expect(toastHandlers.displayErrorToast).toHaveBeenCalledWith(
          "ORG$INVITATION_EXPIRED",
        );
      });

      expect(mockOnClose).toHaveBeenCalledOnce();
      expect(mockOnSuccess).not.toHaveBeenCalled();
    });

    it("should show invalid error toast and call onClose when invitation is invalid", async () => {
      const user = userEvent.setup();

      vi.mocked(organizationService.acceptInvitation).mockRejectedValueOnce(
        createAxiosError(400, "Bad Request", { detail: "invitation_invalid" }),
      );

      renderWithProviders(
        <InvitationAcceptModal
          token={mockToken}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />,
      );

      await user.click(screen.getByTestId("accept-invitation-button"));

      await waitFor(() => {
        expect(toastHandlers.displayErrorToast).toHaveBeenCalledWith(
          "ORG$INVITATION_INVALID",
        );
      });

      expect(mockOnClose).toHaveBeenCalledOnce();
    });

    it("should show already member error toast when user is already a member", async () => {
      const user = userEvent.setup();

      vi.mocked(organizationService.acceptInvitation).mockRejectedValueOnce(
        createAxiosError(409, "Conflict", { detail: "already_member" }),
      );

      renderWithProviders(
        <InvitationAcceptModal
          token={mockToken}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />,
      );

      await user.click(screen.getByTestId("accept-invitation-button"));

      await waitFor(() => {
        expect(toastHandlers.displayErrorToast).toHaveBeenCalledWith(
          "ORG$ALREADY_MEMBER",
        );
      });

      expect(mockOnClose).toHaveBeenCalledOnce();
    });

    it("should show email mismatch error toast when email does not match", async () => {
      const user = userEvent.setup();

      vi.mocked(organizationService.acceptInvitation).mockRejectedValueOnce(
        createAxiosError(403, "Forbidden", { detail: "email_mismatch" }),
      );

      renderWithProviders(
        <InvitationAcceptModal
          token={mockToken}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />,
      );

      await user.click(screen.getByTestId("accept-invitation-button"));

      await waitFor(() => {
        expect(toastHandlers.displayErrorToast).toHaveBeenCalledWith(
          "ORG$INVITATION_EMAIL_MISMATCH",
        );
      });

      expect(mockOnClose).toHaveBeenCalledOnce();
    });

    it("should show generic error toast for unknown errors", async () => {
      const user = userEvent.setup();

      vi.mocked(organizationService.acceptInvitation).mockRejectedValueOnce(
        createAxiosError(500, "Internal Server Error", {
          detail: "unexpected_error",
        }),
      );

      renderWithProviders(
        <InvitationAcceptModal
          token={mockToken}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />,
      );

      await user.click(screen.getByTestId("accept-invitation-button"));

      await waitFor(() => {
        expect(toastHandlers.displayErrorToast).toHaveBeenCalledWith(
          "ORG$INVITATION_ACCEPT_ERROR",
        );
      });

      expect(mockOnClose).toHaveBeenCalledOnce();
    });
  });
});
