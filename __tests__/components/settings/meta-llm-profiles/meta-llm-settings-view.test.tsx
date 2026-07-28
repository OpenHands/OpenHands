import { describe, expect, it, vi, beforeEach, type Mock } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpError } from "@openhands/typescript-client";
import { renderWithProviders } from "test-utils";
import { MetaLlmSettingsView } from "#/components/features/settings/meta-llm-profiles";
import * as useMetaProfilesHook from "#/hooks/query/use-meta-profiles";
import * as useLlmProfilesHook from "#/hooks/query/use-llm-profiles";
import * as useSaveMetaProfileHook from "#/hooks/mutation/use-save-meta-profile";
import * as useActivateMetaProfileHook from "#/hooks/mutation/use-activate-meta-profile";
import * as useDeleteMetaProfileHook from "#/hooks/mutation/use-delete-meta-profile";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";

vi.mock("#/hooks/query/use-meta-profiles");
vi.mock("#/hooks/query/use-llm-profiles");
vi.mock("#/hooks/mutation/use-save-meta-profile");
vi.mock("#/hooks/mutation/use-activate-meta-profile");
vi.mock("#/hooks/mutation/use-delete-meta-profile");
vi.mock("#/api/meta-profiles-service/meta-profiles-service.api");
vi.mock("#/utils/custom-toast-handlers");

const mockMetaProfiles = [
  {
    name: "balanced",
    classifier_model: "minimax",
    default_model: "gpt",
    num_classes: 2,
  },
  {
    name: "cheap",
    classifier_model: "minimax",
    default_model: "deepseek",
    num_classes: 0,
  },
];

const mockLlmProfiles = [
  { name: "minimax", model: "m", base_url: null, api_key_set: true },
  { name: "gpt", model: "g", base_url: null, api_key_set: true },
  { name: "deepseek", model: "d", base_url: null, api_key_set: true },
];

function mockMutation<T>(mutateAsync: Mock, overrides: Partial<T> = {}): T {
  return {
    mutateAsync,
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    isSuccess: false,
    error: null,
    data: undefined,
    reset: vi.fn(),
    status: "idle",
    isIdle: true,
    ...overrides,
  } as T;
}

describe("MetaLlmSettingsView", () => {
  const activateMutateAsync = vi.fn();
  const saveMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useMetaProfilesHook.useMetaProfiles).mockReturnValue({
      data: {
        meta_profiles: mockMetaProfiles,
        active_meta_profile: "balanced",
      },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useMetaProfilesHook.useMetaProfiles>);

    vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue({
      data: { profiles: mockLlmProfiles, active_profile: "minimax" },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useLlmProfilesHook.useLlmProfiles>);

    vi.mocked(useSaveMetaProfileHook.useSaveMetaProfile).mockReturnValue(
      mockMutation(saveMutateAsync),
    );
    vi.mocked(
      useActivateMetaProfileHook.useActivateMetaProfile,
    ).mockReturnValue(mockMutation(activateMutateAsync));
    // The delete hook is consumed by the modal that is always mounted.
    vi.mocked(useDeleteMetaProfileHook.useDeleteMetaProfile).mockReturnValue(
      mockMutation(vi.fn()),
    );
  });

  it("renders the list of meta-profiles with an active badge", () => {
    renderWithProviders(<MetaLlmSettingsView />);

    expect(screen.getByTestId("meta-profile-row-balanced")).toBeInTheDocument();
    expect(screen.getByTestId("meta-profile-row-cheap")).toBeInTheDocument();
    // Only the active one shows the badge
    expect(screen.getAllByTestId("meta-profile-active-badge")).toHaveLength(1);
  });

  it("shows the empty state when there are no meta-profiles", () => {
    vi.mocked(useMetaProfilesHook.useMetaProfiles).mockReturnValue({
      data: { meta_profiles: [], active_meta_profile: null },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useMetaProfilesHook.useMetaProfiles>);

    renderWithProviders(<MetaLlmSettingsView />);

    expect(screen.getByTestId("meta-profile-empty")).toBeInTheDocument();
  });

  it("hints when there are no LLM profiles to route between", () => {
    vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue({
      data: { profiles: [], active_profile: null },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useLlmProfilesHook.useLlmProfiles>);

    renderWithProviders(<MetaLlmSettingsView />);

    expect(
      screen.getByTestId("meta-profile-no-llm-profiles"),
    ).toBeInTheDocument();
  });

  it("opens the editor when clicking Add meta-profile", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MetaLlmSettingsView />);

    await user.click(screen.getByTestId("add-meta-profile"));

    expect(screen.getByTestId("meta-profile-editor")).toBeInTheDocument();
    expect(screen.getByTestId("meta-profile-name-input")).toBeInTheDocument();
  });

  it("activates a meta-profile via the actions menu", async () => {
    const user = userEvent.setup();
    activateMutateAsync.mockResolvedValue({ name: "cheap" });
    renderWithProviders(<MetaLlmSettingsView />);

    await user.click(screen.getByTestId("meta-profile-menu-trigger-cheap"));
    await user.click(screen.getByTestId("meta-profile-set-active"));

    await waitFor(() =>
      expect(activateMutateAsync).toHaveBeenCalledWith("cheap"),
    );
  });

  it("loads the config and opens the editor via the actions menu", async () => {
    const user = userEvent.setup();
    vi.mocked(MetaProfilesService.getMetaProfile).mockResolvedValue({
      name: "balanced",
      config: {
        classifier_model: "minimax",
        default_model: "gpt",
        classes: [{ description: "UI", model: "deepseek" }],
      },
    });
    renderWithProviders(<MetaLlmSettingsView />);

    await user.click(screen.getByTestId("meta-profile-menu-trigger-balanced"));
    await user.click(screen.getByTestId("meta-profile-edit"));

    await waitFor(() =>
      expect(screen.getByTestId("meta-profile-editor")).toBeInTheDocument(),
    );
    expect(MetaProfilesService.getMetaProfile).toHaveBeenCalledWith("balanced");
  });

  it("shows an explicit unsupported-backend message when the API is missing (404)", () => {
    // Older backends (pre software-agent-sdk #3744) have no /api/meta-profiles
    // endpoint and return 404; the page must explain that instead of a dead
    // generic error, and must not offer Add.
    vi.mocked(useMetaProfilesHook.useMetaProfiles).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new HttpError(404, "Not Found"),
    } as unknown as ReturnType<typeof useMetaProfilesHook.useMetaProfiles>);

    renderWithProviders(<MetaLlmSettingsView />);

    expect(screen.getByTestId("meta-profile-unsupported")).toBeInTheDocument();
    expect(screen.queryByTestId("add-meta-profile")).not.toBeInTheDocument();
  });

  it("shows the generic error for non-404 failures", () => {
    vi.mocked(useMetaProfilesHook.useMetaProfiles).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new HttpError(500, "Internal Server Error"),
    } as unknown as ReturnType<typeof useMetaProfilesHook.useMetaProfiles>);

    renderWithProviders(<MetaLlmSettingsView />);

    expect(
      screen.queryByTestId("meta-profile-unsupported"),
    ).not.toBeInTheDocument();
    // The Add affordance remains for transient/server errors.
    expect(screen.getByTestId("add-meta-profile")).toBeInTheDocument();
  });

  it("disables Set active in the menu for the already-active profile", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MetaLlmSettingsView />);

    await user.click(screen.getByTestId("meta-profile-menu-trigger-balanced"));

    expect(screen.getByTestId("meta-profile-set-active")).toBeDisabled();
  });
});
