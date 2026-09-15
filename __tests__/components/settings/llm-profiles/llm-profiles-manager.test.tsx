import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LlmProfilesManager } from "#/components/features/settings/llm-profiles/llm-profiles-manager";
import ProfilesService, {
  ProfileInfo,
} from "#/api/profiles-service/profiles-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        SETTINGS$AVAILABLE_PROFILES: "Available LLM Profiles",
        SETTINGS$ADD_LLM_PROFILE: "Add LLM Profile",
        SETTINGS$PROFILES_LOAD_ERROR: "Failed to load profiles",
        SETTINGS$PROFILES_EMPTY: "No profiles saved yet",
        SETTINGS$PROFILE_ACTIVE: "Active",
        SETTINGS$PROFILE_MENU: "Profile menu",
        SETTINGS$PROFILE_EDIT: "Edit",
        SETTINGS$PROFILE_SET_ACTIVE: "Set as active",
        SETTINGS$PROFILE_SET_DEFAULT: "Set as default",
        SETTINGS$PROFILE_DEFAULT: "Default",
        SETTINGS$PROFILE_RENAME_TITLE: "Rename Profile",
        SETTINGS$PROFILE_DELETE_TITLE: "Delete Profile",
        SETTINGS$PROFILE_DELETE_CONFIRMATION: params?.name
          ? `Are you sure you want to delete "${params.name}"?`
          : "Are you sure you want to delete this profile?",
        SETTINGS$PROFILE_ACTIVATED: params?.name
          ? `Profile "${params.name}" activated`
          : "Profile activated",
        SETTINGS$PROFILE_DUPLICATED: params?.name
          ? `Profile "${params.name}" duplicated`
          : "Profile duplicated",
        SETTINGS$PROFILE_NAME_LABEL: "Profile Name",
        SETTINGS$PROFILE_NAME_PLACEHOLDER: "Enter profile name",
        SETTINGS$PROFILE_NAME_RULE:
          "1-64 chars, start with alphanumeric, then alphanumerics or . _ -",
        BUTTON$RENAME: "Rename",
        BUTTON$DUPLICATE: "Duplicate",
        BUTTON$DELETE: "Delete",
        BUTTON$CANCEL: "Cancel",
        ERROR$GENERIC: "An error occurred",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("#/api/profiles-service/profiles-service.api");
vi.mock("#/utils/custom-toast-handlers");

const canManage = vi.hoisted(() => ({ value: true }));
vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => canManage.value,
}));

function makeProfile(overrides: Partial<ProfileInfo> = {}): ProfileInfo {
  return {
    name: "gpt-4-profile",
    model: "openai/gpt-4",
    base_url: null,
    api_key_set: true,
    ...overrides,
  };
}

const mockProfiles: ProfileInfo[] = [
  makeProfile(),
  makeProfile({
    name: "claude-profile",
    model: "anthropic/claude-3",
    base_url: "https://api.anthropic.com",
    api_key_set: false,
  }),
];

describe("LlmProfilesManager", () => {
  const renderManager = (
    props: {
      onAddProfile?: () => void;
      onEditProfile?: (profile: ProfileInfo) => void;
    } = {},
    options: { canManage?: boolean } = {},
  ) => {
    vi.clearAllMocks();
    canManage.value = options.canManage ?? true;
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    return render(
      <QueryClientProvider client={queryClient}>
        <LlmProfilesManager {...props} />
      </QueryClientProvider>,
    );
  };

  it("displays the section title", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager();

    expect(screen.getByText("Available LLM Profiles")).toBeInTheDocument();
  });

  it("shows Add LLM Profile button when onAddProfile is provided", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    renderManager({ onAddProfile: vi.fn() });

    expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
    expect(screen.getByText("Add LLM Profile")).toBeInTheDocument();
  });

  it("does not show Add LLM Profile button when onAddProfile is not provided", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    renderManager();

    expect(screen.queryByTestId("add-llm-profile")).not.toBeInTheDocument();
  });

  it("calls onAddProfile when Add LLM Profile is clicked", async () => {
    const user = userEvent.setup();
    const handleAddProfile = vi.fn();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    renderManager({ onAddProfile: handleAddProfile });

    await user.click(screen.getByTestId("add-llm-profile"));

    expect(handleAddProfile).toHaveBeenCalledTimes(1);
  });

  it("hides profile mutation controls from view-only members", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager(
      { onAddProfile: vi.fn(), onEditProfile: vi.fn() },
      { canManage: false },
    );

    await screen.findByText("gpt-4-profile");
    expect(screen.queryByTestId("add-llm-profile")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("profile-menu-trigger"),
    ).not.toBeInTheDocument();
  });

  it("displays profiles when they exist", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager();

    await screen.findByText("gpt-4-profile");
    expect(screen.getByText("claude-profile")).toBeInTheDocument();

    const profileRows = screen.getAllByTestId("profile-row");
    expect(
      within(profileRows[0]).getByTestId("profile-active-badge"),
    ).toHaveTextContent("Default");
    expect(
      within(profileRows[1]).queryByTestId("profile-active-badge"),
    ).not.toBeInTheDocument();
  });

  it("shows empty state when no profiles exist", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    renderManager();

    await screen.findByText("No profiles saved yet");
  });

  it("shows loading spinner while loading", () => {
    // Mock a never-resolving promise to keep loading state
    vi.mocked(ProfilesService.listProfiles).mockImplementation(
      () => new Promise(() => {}),
    );

    renderManager();

    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("shows error message when loading fails", async () => {
    vi.mocked(ProfilesService.listProfiles).mockRejectedValue(
      new Error("Network error"),
    );

    renderManager();

    await screen.findByText("Failed to load profiles");
  });

  it("calls onEditProfile when Edit is clicked from profile menu", async () => {
    const user = userEvent.setup();
    const handleEditProfile = vi.fn();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager({ onEditProfile: handleEditProfile });

    // Wait for profiles to load
    await screen.findByText("gpt-4-profile");

    // Click the first profile's menu trigger
    const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
    await user.click(menuTriggers[0]);

    // Click Edit
    await user.click(screen.getByText("Edit"));

    expect(handleEditProfile).toHaveBeenCalledWith(mockProfiles[0]);
  });

  it("closes the Edit menu safely when no edit callback is provided", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager();
    await screen.findByText("gpt-4-profile");
    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);

    await user.click(screen.getByTestId("profile-edit"));

    expect(screen.queryByTestId("profile-edit")).not.toBeInTheDocument();
    expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();
  });

  it("activates a profile and confirms the selected profile to the user", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });
    vi.mocked(ProfilesService.activateProfile).mockResolvedValue({
      name: "claude-profile",
      message: "Profile activated",
      llm_applied: true,
    });
    const user = userEvent.setup();

    renderManager();
    await screen.findByText("claude-profile");
    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-set-active"));

    await waitFor(() => {
      expect(ProfilesService.activateProfile).toHaveBeenCalledWith(
        "claude-profile",
      );
    });
    await waitFor(() => {
      expect(displaySuccessToast).toHaveBeenCalledWith(
        'Profile "claude-profile" activated',
      );
    });
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("reports activation failures without claiming success", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });
    const failure = new Error("activation unavailable");
    vi.mocked(ProfilesService.activateProfile).mockRejectedValue(failure);
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const user = userEvent.setup();

    renderManager();
    await screen.findByText("claude-profile");
    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-set-active"));

    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith("An error occurred");
    });
    expect(displaySuccessToast).not.toHaveBeenCalled();
    expect(consoleError).toHaveBeenCalledWith(
      "Failed to activate profile:",
      failure,
    );
    consoleError.mockRestore();
  });

  it("duplicates a profile with encrypted secrets under the first available name", async () => {
    const profiles = [
      ...mockProfiles,
      makeProfile({ name: "gpt-4-profile-copy" }),
      makeProfile({ name: "gpt-4-profile-copy-1" }),
    ];
    const llm = {
      model: "openai/gpt-4.1",
      api_key: "encrypted:profile-key",
      base_url: "https://api.openai.com/v1",
    };
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles,
      active_profile: "gpt-4-profile",
    });
    vi.mocked(ProfilesService.getProfile).mockResolvedValue({
      name: "gpt-4-profile",
      api_key_set: true,
      config: llm,
    });
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "gpt-4-profile-copy-2",
      message: "Profile saved",
    });
    const user = userEvent.setup();

    renderManager();
    await screen.findByText("gpt-4-profile");
    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-duplicate"));

    await waitFor(() => {
      expect(ProfilesService.getProfile).toHaveBeenCalledWith(
        "gpt-4-profile",
        "encrypted",
      );
    });
    await waitFor(() => {
      expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
        "gpt-4-profile-copy-2",
        {
          llm,
          include_secrets: true,
        },
      );
    });
    expect(displaySuccessToast).toHaveBeenCalledWith(
      'Profile "gpt-4-profile-copy-2" duplicated',
    );
  });

  it("uses the simple copy suffix when it is available", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });
    vi.mocked(ProfilesService.getProfile).mockResolvedValue({
      name: "gpt-4-profile",
      api_key_set: false,
      config: { model: "openai/gpt-4" },
    });
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "gpt-4-profile-copy",
      message: "Profile saved",
    });
    const user = userEvent.setup();

    renderManager();
    await screen.findByText("gpt-4-profile");
    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-duplicate"));

    await waitFor(() => {
      expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
        "gpt-4-profile-copy",
        expect.objectContaining({ include_secrets: true }),
      );
    });
  });

  it("reports duplicate failures without claiming a profile was created", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });
    const failure = new Error("profile detail unavailable");
    vi.mocked(ProfilesService.getProfile).mockRejectedValue(failure);
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const user = userEvent.setup();

    renderManager();
    await screen.findByText("gpt-4-profile");
    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-duplicate"));

    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith("An error occurred");
    });
    expect(ProfilesService.saveProfile).not.toHaveBeenCalled();
    expect(displaySuccessToast).not.toHaveBeenCalled();
    expect(consoleError).toHaveBeenCalledWith(
      "Failed to duplicate profile:",
      failure,
    );
    consoleError.mockRestore();
  });

  it("opens rename modal when Rename is clicked from profile menu", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager();

    await screen.findByText("gpt-4-profile");

    const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
    await user.click(menuTriggers[0]);
    await user.click(screen.getByText("Rename"));

    // Rename modal should appear with input pre-filled
    expect(screen.getByTestId("rename-profile-input")).toHaveValue(
      "gpt-4-profile",
    );
  });

  it("opens delete modal when Delete is clicked from profile menu", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager();

    await screen.findByText("claude-profile");

    // Use the second profile (claude-profile) — the active profile's Delete is disabled
    const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
    await user.click(menuTriggers[1]);
    await user.click(screen.getByText("Delete"));

    // Delete modal should appear with confirmation message
    expect(
      screen.getByText('Are you sure you want to delete "claude-profile"?'),
    ).toBeInTheDocument();
  });

  it("closes rename modal when onClose is called", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager();

    await screen.findByText("gpt-4-profile");

    // Open rename modal
    const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
    await user.click(menuTriggers[0]);
    await user.click(screen.getByText("Rename"));

    expect(screen.getByTestId("rename-profile-input")).toBeInTheDocument();

    // Click Cancel
    await user.click(screen.getByText("Cancel"));

    // Modal should be closed
    expect(
      screen.queryByTestId("rename-profile-input"),
    ).not.toBeInTheDocument();
  });

  it("closes delete modal when onClose is called", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_profile: "gpt-4-profile",
    });

    renderManager();

    await screen.findByText("claude-profile");

    // Open delete modal — use the second profile (claude-profile); active profile's Delete is disabled
    const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
    await user.click(menuTriggers[1]);
    await user.click(screen.getByText("Delete"));

    expect(
      screen.getByText('Are you sure you want to delete "claude-profile"?'),
    ).toBeInTheDocument();

    // Click Cancel
    await user.click(screen.getByText("Cancel"));

    // Modal should be closed
    expect(
      screen.queryByText('Are you sure you want to delete "claude-profile"?'),
    ).not.toBeInTheDocument();
  });
});
