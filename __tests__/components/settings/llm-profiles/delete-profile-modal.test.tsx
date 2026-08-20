import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DeleteProfileModal } from "#/components/features/settings/llm-profiles/delete-profile-modal";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import * as toastHandlers from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        "SETTINGS$PROFILE_DELETE_TITLE": "Delete Profile",
        "SETTINGS$PROFILE_DELETE_CONFIRMATION": params?.name
          ? `Are you sure you want to delete "${params.name}"?`
          : "Are you sure you want to delete this profile?",
        "SETTINGS$PROFILE_DELETED": params?.name
          ? `Profile "${params.name}" deleted`
          : "Profile deleted",
        "BUTTON$DELETE": "Delete",
        "BUTTON$CANCEL": "Cancel",
        "ERROR$GENERIC": "An error occurred",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("#/api/profiles-service/profiles-service.api");
vi.mock("#/utils/custom-toast-handlers");

const mockProfile: ProfileInfo = {
  name: "profile-to-delete",
  model: "openai/gpt-4",
  base_url: null,
  api_key_set: true,
};

describe("DeleteProfileModal", () => {
  let queryClient: QueryClient;

  const renderModal = (
    profile: ProfileInfo | null,
    onClose = vi.fn(),
  ) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <DeleteProfileModal profile={profile} onClose={onClose} />
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("returns null when profile is null", () => {
    const { container } = renderModal(null);
    expect(container.firstChild).toBeNull();
  });

  it("renders the modal with confirmation message", () => {
    renderModal(mockProfile);
    expect(
      screen.getByText('Are you sure you want to delete "profile-to-delete"?'),
    ).toBeInTheDocument();
  });

  it("shows Delete and Cancel buttons", () => {
    renderModal(mockProfile);
    expect(screen.getByTestId("delete-profile-confirm")).toHaveTextContent(
      "Delete",
    );
    expect(screen.getByText("Cancel")).toBeInTheDocument();
  });

  it("places Cancel before Delete in the footer so the dominant action is the last focusable button", () => {
    // Arrange: render the modal so both footer buttons are mounted.
    renderModal(mockProfile);

    // Act: locate both footer buttons.
    const cancel = screen.getByText("Cancel");
    const danger = screen.getByTestId("delete-profile-confirm");

    // Assert: Cancel precedes the dominant Delete action in DOM order.
    // eslint-disable-next-line no-bitwise
    expect(
      cancel.compareDocumentPosition(danger) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("calls onClose when Cancel is clicked", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();
    renderModal(mockProfile, handleClose);

    await user.click(screen.getByText("Cancel"));

    expect(handleClose).toHaveBeenCalledTimes(1);
    expect(ProfilesService.deleteProfile).not.toHaveBeenCalled();
  });

  it("calls deleteProfile and shows success toast on successful delete", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();
    vi.mocked(ProfilesService.deleteProfile).mockResolvedValue({
      name: "profile-to-delete",
      message: "Profile deleted",
    });

    renderModal(mockProfile, handleClose);

    await user.click(screen.getByTestId("delete-profile-confirm"));

    await waitFor(() => {
      expect(ProfilesService.deleteProfile).toHaveBeenCalledWith(
        "profile-to-delete",
      );
    });

    expect(toastHandlers.displaySuccessToast).toHaveBeenCalledWith(
      'Profile "profile-to-delete" deleted',
    );
    expect(handleClose).toHaveBeenCalled();
  });

  it("shows error toast on delete failure", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();
    vi.mocked(ProfilesService.deleteProfile).mockRejectedValue(
      new Error("Delete failed"),
    );

    renderModal(mockProfile, handleClose);

    await user.click(screen.getByTestId("delete-profile-confirm"));

    await waitFor(() => {
      expect(toastHandlers.displayErrorToast).toHaveBeenCalledWith(
        "Delete failed",
      );
    });

    expect(handleClose).not.toHaveBeenCalled();
  });

  it("shows generic error message for non-Error exceptions", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.deleteProfile).mockRejectedValue("Unknown error");

    renderModal(mockProfile);

    await user.click(screen.getByTestId("delete-profile-confirm"));

    await waitFor(() => {
      expect(toastHandlers.displayErrorToast).toHaveBeenCalledWith(
        "An error occurred",
      );
    });
  });

  it("delete button has danger variant styling", () => {
    renderModal(mockProfile);
    const deleteButton = screen.getByTestId("delete-profile-confirm");
    // The BrandButton with variant="danger" should be rendered
    expect(deleteButton).toBeInTheDocument();
  });

  describe("isPending state", () => {
    it("prevents closing modal during deletion", async () => {
      const user = userEvent.setup();
      vi.mocked(ProfilesService.deleteProfile).mockImplementation(
        () => new Promise(() => {}), // Never resolves
      );
      const handleClose = vi.fn();
      renderModal(mockProfile, handleClose);

      await user.click(screen.getByTestId("delete-profile-confirm"));
      await user.click(screen.getByText("Cancel"));

      expect(handleClose).not.toHaveBeenCalled();
    });

    it("disables Cancel button during deletion", async () => {
      const user = userEvent.setup();
      vi.mocked(ProfilesService.deleteProfile).mockImplementation(
        () => new Promise(() => {}), // Never resolves
      );
      renderModal(mockProfile);

      await user.click(screen.getByTestId("delete-profile-confirm"));

      expect(screen.getByText("Cancel")).toBeDisabled();
    });

    it("disables Delete button during deletion", async () => {
      const user = userEvent.setup();
      vi.mocked(ProfilesService.deleteProfile).mockImplementation(
        () => new Promise(() => {}), // Never resolves
      );
      renderModal(mockProfile);

      await user.click(screen.getByTestId("delete-profile-confirm"));

      expect(screen.getByTestId("delete-profile-confirm")).toBeDisabled();
    });
  });
});
