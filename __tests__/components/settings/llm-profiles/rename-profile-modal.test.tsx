import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RenameProfileModal } from "#/components/features/settings/llm-profiles/rename-profile-modal";
import { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        "SETTINGS$PROFILE_RENAME_TITLE": "Rename Profile",
        "SETTINGS$PROFILE_NAME_LABEL": "Profile Name",
        "SETTINGS$PROFILE_NAME_PLACEHOLDER": "Enter profile name",
        "SETTINGS$PROFILE_NAME_RULE":
          "1-64 chars, start with alphanumeric, then alphanumerics or . _ -",
        "SETTINGS$PROFILE_RENAMED": params?.name
          ? `Profile renamed to ${params.name}`
          : "Profile renamed",
        "BUTTON$RENAME": "Rename",
        "BUTTON$CANCEL": "Cancel",
        "ERROR$GENERIC": "An error occurred",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("#/api/profiles-service/profiles-service.api");

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

const mockProfile: ProfileInfo = {
  name: "old-profile-name",
  model: "openai/gpt-4",
  base_url: null,
  api_key_set: true,
};

describe("RenameProfileModal", () => {
  let queryClient: QueryClient;

  const renderModal = (profile: ProfileInfo | null, onClose = vi.fn()) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <RenameProfileModal profile={profile} onClose={onClose} />
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

  it("renders the modal when profile is provided", () => {
    renderModal(mockProfile);
    expect(screen.getByTestId("rename-profile-modal")).toBeInTheDocument();
  });

  it("pre-fills the input with the current profile name", () => {
    renderModal(mockProfile);
    expect(screen.getByTestId("rename-profile-input")).toHaveValue(
      "old-profile-name",
    );
  });

  it("shows Rename and Cancel buttons", () => {
    renderModal(mockProfile);
    expect(screen.getByTestId("rename-profile-submit")).toHaveTextContent(
      "Rename",
    );
    expect(screen.getByText("Cancel")).toBeInTheDocument();
  });

  it("places Cancel before Rename in the footer so the dominant action is the last focusable button", () => {
    // Arrange: render the modal so both footer buttons are mounted.
    renderModal(mockProfile);

    // Act: locate both footer buttons.
    const cancel = screen.getByText("Cancel");
    const submit = screen.getByTestId("rename-profile-submit");

    // Assert: Cancel precedes the dominant Rename action in DOM order.
    // eslint-disable-next-line no-bitwise
    expect(
      cancel.compareDocumentPosition(submit) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("calls onClose when Cancel is clicked", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();
    renderModal(mockProfile, handleClose);

    await user.click(screen.getByText("Cancel"));

    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when name is unchanged and Rename is clicked", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();
    renderModal(mockProfile, handleClose);

    // Name is already "old-profile-name", so clicking Rename should just close
    await user.click(screen.getByTestId("rename-profile-submit"));

    expect(handleClose).toHaveBeenCalledTimes(1);
    expect(ProfilesService.renameProfile).not.toHaveBeenCalled();
  });

  it("calls renameProfile and closes on successful rename", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
      name: "new-profile-name",
      message: "Profile renamed",
    });

    renderModal(mockProfile, handleClose);

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "new-profile-name");
    await user.click(screen.getByTestId("rename-profile-submit"));

    await waitFor(() => {
      expect(ProfilesService.renameProfile).toHaveBeenCalledWith(
        "old-profile-name",
        "new-profile-name",
      );
    });

    await waitFor(() => {
      expect(handleClose).toHaveBeenCalled();
    });
  });

  it("disables submit button when name is invalid", async () => {
    const user = userEvent.setup();
    renderModal(mockProfile);

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, ".invalid-name");

    const submitButton = screen.getByTestId("rename-profile-submit");
    expect(submitButton).toBeDisabled();
  });

  it("marks the input as invalid for names violating the format rule", async () => {
    const user = userEvent.setup();
    renderModal(mockProfile);

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, ".invalid-name");

    expect(input).toHaveAttribute("aria-invalid", "true");
  });

  it("updates input value when a different profile is passed", () => {
    const { rerender } = render(
      <QueryClientProvider client={queryClient}>
        <RenameProfileModal profile={mockProfile} onClose={vi.fn()} />
      </QueryClientProvider>,
    );

    expect(screen.getByTestId("rename-profile-input")).toHaveValue(
      "old-profile-name",
    );

    const newProfile: ProfileInfo = {
      ...mockProfile,
      name: "different-profile",
    };

    rerender(
      <QueryClientProvider client={queryClient}>
        <RenameProfileModal profile={newProfile} onClose={vi.fn()} />
      </QueryClientProvider>,
    );

    expect(screen.getByTestId("rename-profile-input")).toHaveValue(
      "different-profile",
    );
  });

  it("shows error toast and keeps modal open on rename failure", async () => {
    const user = userEvent.setup();
    const { displayErrorToast } = await import(
      "#/utils/custom-toast-handlers"
    );
    vi.mocked(ProfilesService.renameProfile).mockRejectedValue(
      new Error("Name already exists"),
    );

    const onClose = vi.fn();
    renderModal(mockProfile, onClose);

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "duplicate-name");
    await user.click(screen.getByTestId("rename-profile-submit"));

    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith("Name already exists");
    });

    // Modal should stay open after error (onClose not called)
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByTestId("rename-profile-modal")).toBeInTheDocument();
  });

  it("submits form when Enter key is pressed with valid name", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();
    vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
      name: "new-name",
      message: "Profile renamed",
    });

    renderModal(mockProfile, handleClose);

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "new-name{Enter}");

    await waitFor(() => {
      expect(ProfilesService.renameProfile).toHaveBeenCalledWith(
        "old-profile-name",
        "new-name",
      );
    });

    await waitFor(() => {
      expect(handleClose).toHaveBeenCalled();
    });
  });

  it("does not submit form when Enter key is pressed with invalid name", async () => {
    const user = userEvent.setup();
    const handleClose = vi.fn();

    renderModal(mockProfile, handleClose);

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, ".invalid{Enter}");

    // Should not call rename or close
    expect(ProfilesService.renameProfile).not.toHaveBeenCalled();
    expect(handleClose).not.toHaveBeenCalled();
    expect(screen.getByTestId("rename-profile-modal")).toBeInTheDocument();
  });
});
