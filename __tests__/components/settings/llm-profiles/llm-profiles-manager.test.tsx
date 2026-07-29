import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LlmProfilesManager } from "#/components/features/settings/llm-profiles/llm-profiles-manager";
import ProfilesService, {
  ProfileInfo,
} from "#/api/profiles-service/profiles-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        "SETTINGS$AVAILABLE_PROFILES": "Available LLM Profiles",
        "SETTINGS$ADD_LLM_PROFILE": "Add LLM Profile",
        "SETTINGS$PROFILES_LOAD_ERROR": "Failed to load profiles",
        "SETTINGS$PROFILES_EMPTY": "No profiles saved yet",
        "SETTINGS$PROFILE_ACTIVE": "Active",
        "SETTINGS$PROFILE_MENU": "Profile menu",
        "SETTINGS$PROFILE_EDIT": "Edit",
        "SETTINGS$PROFILE_SET_ACTIVE": "Set as active",
        "SETTINGS$PROFILE_RENAME_TITLE": "Rename Profile",
        "SETTINGS$PROFILE_DELETE_TITLE": "Delete Profile",
        "SETTINGS$PROFILE_DELETE_CONFIRMATION": params?.name
          ? `Are you sure you want to delete "${params.name}"?`
          : "Are you sure you want to delete this profile?",
        "SETTINGS$PROFILE_ACTIVATED": params?.name
          ? `Profile "${params.name}" activated`
          : "Profile activated",
        "SETTINGS$PROFILE_NAME_LABEL": "Profile Name",
        "SETTINGS$PROFILE_NAME_PLACEHOLDER": "Enter profile name",
        "SETTINGS$PROFILE_NAME_RULE":
          "1-64 chars, start with alphanumeric, then alphanumerics or . _ -",
        "BUTTON$RENAME": "Rename",
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

const mockProfiles: ProfileInfo[] = [
  {
    name: "gpt-4-profile",
    model: "openai/gpt-4",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "claude-profile",
    model: "anthropic/claude-3",
    base_url: "https://api.anthropic.com",
    api_key_set: false,
  },
];

describe("LlmProfilesManager", () => {
  let queryClient: QueryClient;

  const renderManager = (props: {
    onAddProfile?: () => void;
    onEditProfile?: (profile: ProfileInfo) => void;
  } = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <LlmProfilesManager {...props} />
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

  it("displays the section title", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles, active_profile: "gpt-4-profile",
    });

    renderManager();

    expect(screen.getByText("Available LLM Profiles")).toBeInTheDocument();
  });

  it("shows Add LLM Profile button when onAddProfile is provided", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [], active_profile: null,
    });

    renderManager({ onAddProfile: vi.fn() });

    expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
    expect(screen.getByText("Add LLM Profile")).toBeInTheDocument();
  });

  it("does not show Add LLM Profile button when onAddProfile is not provided", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [], active_profile: null,
    });

    renderManager();

    expect(screen.queryByTestId("add-llm-profile")).not.toBeInTheDocument();
  });

  it("calls onAddProfile when Add LLM Profile is clicked", async () => {
    const user = userEvent.setup();
    const handleAddProfile = vi.fn();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [], active_profile: null,
    });

    renderManager({ onAddProfile: handleAddProfile });

    await user.click(screen.getByTestId("add-llm-profile"));

    expect(handleAddProfile).toHaveBeenCalledTimes(1);
  });

  it("displays profiles when they exist", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles, active_profile: "gpt-4-profile",
    });

    renderManager();

    await screen.findByText("gpt-4-profile");
    expect(screen.getByText("claude-profile")).toBeInTheDocument();
  });

  it("shows empty state when no profiles exist", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [], active_profile: null,
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
      profiles: mockProfiles, active_profile: "gpt-4-profile",
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

  it("opens rename modal when Rename is clicked from profile menu", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles, active_profile: "gpt-4-profile",
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
      profiles: mockProfiles, active_profile: "gpt-4-profile",
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
      profiles: mockProfiles, active_profile: "gpt-4-profile",
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
    expect(screen.queryByTestId("rename-profile-input")).not.toBeInTheDocument();
  });

  it("closes delete modal when onClose is called", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles, active_profile: "gpt-4-profile",
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
