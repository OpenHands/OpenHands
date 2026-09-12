import { describe, expect, it, vi, beforeEach, type Mock } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpError } from "@openhands/typescript-client";
import { renderWithProviders } from "test-utils";
import { MetaLlmSettingsView } from "#/components/features/settings/meta-llm-profiles";
import * as useMetaProfilesHook from "#/hooks/query/use-meta-profiles";
import * as useLlmProfilesHook from "#/hooks/query/use-llm-profiles";
import * as useProviderConnectionsHook from "#/hooks/query/use-provider-connections";
import * as useSaveMetaProfileHook from "#/hooks/mutation/use-save-meta-profile";
import * as useActivateMetaProfileHook from "#/hooks/mutation/use-activate-meta-profile";
import * as useDeleteMetaProfileHook from "#/hooks/mutation/use-delete-meta-profile";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import {
  DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT,
  DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME,
  DEFAULT_MIN_COST_PARETO_META_PROFILE_DEFAULT,
  DEFAULT_MIN_COST_PARETO_META_PROFILE_NAME,
} from "#/components/features/settings/meta-llm-profiles/default-meta-profile";
import { collectRequiredRouterModelNames } from "#/components/features/settings/meta-llm-profiles/router-profiles";

vi.mock("#/hooks/query/use-meta-profiles");
vi.mock("#/hooks/query/use-llm-profiles");
vi.mock("#/hooks/query/use-provider-connections");
vi.mock("#/hooks/mutation/use-save-meta-profile");
vi.mock("#/hooks/mutation/use-activate-meta-profile");
vi.mock("#/hooks/mutation/use-delete-meta-profile");
vi.mock("#/api/meta-profiles-service/meta-profiles-service.api");
vi.mock("#/api/profiles-service/profiles-service.api");
vi.mock("#/utils/custom-toast-handlers");

const mockMetaProfiles = [
  {
    name: "balanced",
    classifier_model: "minimax",
    default_model: "gpt",
    num_classes: 0,
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

const mockProviderConnections = [
  {
    id: "conn-openhands",
    display_name: "OpenHands",
    provider: "openhands",
    base_url: null,
    created_at: 0,
    updated_at: 0,
    api_key_set: true,
  },
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

    vi.mocked(
      useProviderConnectionsHook.useProviderConnections,
    ).mockReturnValue({
      data: mockProviderConnections,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<
      typeof useProviderConnectionsHook.useProviderConnections
    >);

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
    vi.mocked(ProfilesService.getProfile).mockResolvedValue({
      name: "minimax",
      api_key_set: true,
      config: {
        model: "litellm_proxy/template",
        base_url: "https://llm-proxy.example",
        api_key: "gAAAA_encrypted",
      },
    });
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "created-profile",
      message: "Profile saved",
    });
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

  const openMaxScoreTemplate = async (
    user: ReturnType<typeof userEvent.setup>,
  ) => {
    await user.click(screen.getByTestId("add-meta-profile"));
    await user.click(screen.getByTestId("meta-profile-template-max-score"));
  };

  it("opens the template chooser when clicking Add Model Router", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MetaLlmSettingsView />);

    await user.click(screen.getByTestId("add-meta-profile"));

    expect(
      screen.getByTestId("meta-profile-template-modal"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("meta-profile-template-max-score")).toBeEnabled();
    expect(screen.getByTestId("meta-profile-template-min-cost")).toBeEnabled();
    expect(screen.getByTestId("meta-profile-template-custom")).toBeEnabled();
  });

  it("opens the max-score default editor from the template chooser", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MetaLlmSettingsView />);

    await openMaxScoreTemplate(user);

    expect(screen.getByTestId("meta-profile-editor")).toBeInTheDocument();
    expect(screen.getByTestId("meta-profile-name-input")).toHaveValue(
      DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME,
    );
  });

  it("opens the min-cost default editor from the template chooser", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MetaLlmSettingsView />);

    await user.click(screen.getByTestId("add-meta-profile"));
    await user.click(screen.getByTestId("meta-profile-template-min-cost"));

    expect(screen.getByTestId("meta-profile-name-input")).toHaveValue(
      DEFAULT_MIN_COST_PARETO_META_PROFILE_NAME,
    );
    expect(screen.getByTestId("meta-profile-prompt-template")).toHaveValue(
      DEFAULT_MIN_COST_PARETO_META_PROFILE_DEFAULT.prompt_template,
    );
    expect(screen.getByTestId("meta-profile-model-table")).toHaveValue(
      DEFAULT_MIN_COST_PARETO_META_PROFILE_DEFAULT.model_table,
    );
    // The built-in templates pre-select the first provider connection so the
    // router's LLM profiles are created on save.
    expect(screen.getByTestId("meta-profile-router-connection")).toHaveValue(
      "OpenHands (openhands)",
    );
  });

  it("opens a blank custom editor from the template chooser", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MetaLlmSettingsView />);

    await user.click(screen.getByTestId("add-meta-profile"));
    await user.click(screen.getByTestId("meta-profile-template-custom"));

    expect(screen.getByTestId("meta-profile-name-input")).toHaveValue("");
    expect(screen.getByTestId("meta-profile-classifier-input")).toHaveValue("");
    expect(screen.getByTestId("meta-profile-default-input")).toHaveValue("");
    expect(screen.getByTestId("meta-profile-prompt-template")).toHaveValue("");
    expect(screen.getByTestId("meta-profile-model-table")).toHaveValue("");
    // Custom profiles default to "don't create profiles".
    expect(
      screen.getByTestId("meta-profile-router-connection"),
    ).not.toHaveValue("OpenHands (openhands)");
  });

  it("creates missing router LLM profiles linked to the selected provider connection", async () => {
    const user = userEvent.setup();
    saveMutateAsync.mockResolvedValue({ name: "default-max-score-pareto" });
    renderWithProviders(<MetaLlmSettingsView />);

    await openMaxScoreTemplate(user);
    await user.click(screen.getByTestId("meta-profile-save"));

    // Every model in the built-in table (plus classifier/default) that is not
    // already a saved profile is created, linked to the connection.
    const expectedNames = collectRequiredRouterModelNames(
      DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT,
    ).filter((n) => !["minimax", "gpt", "deepseek"].includes(n.toLowerCase()));

    await waitFor(() =>
      expect(ProfilesService.saveProfile).toHaveBeenCalledWith("GPT-5.4", {
        llm: {
          model: "openhands/GPT-5.4",
          usage_id: "GPT-5.4",
          provider_connection_id: "conn-openhands",
        },
        include_secrets: true,
      }),
    );
    expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(
      expectedNames.length,
    );
    // Router profiles reuse shared credentials via the connection, so they
    // must not clone an active profile's key.
    expect(ProfilesService.getProfile).not.toHaveBeenCalled();
    await waitFor(() => expect(saveMutateAsync).toHaveBeenCalled());
  });

  it("activates the first meta-profile after creating it", async () => {
    const user = userEvent.setup();
    vi.mocked(useMetaProfilesHook.useMetaProfiles).mockReturnValue({
      data: { meta_profiles: [], active_meta_profile: null },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useMetaProfilesHook.useMetaProfiles>);
    saveMutateAsync.mockResolvedValue({ name: "pareto" });
    activateMutateAsync.mockResolvedValue({ name: "pareto" });
    renderWithProviders(<MetaLlmSettingsView />);

    await openMaxScoreTemplate(user);
    await user.clear(screen.getByTestId("meta-profile-name-input"));
    await user.type(screen.getByTestId("meta-profile-name-input"), "pareto");
    fireEvent.change(screen.getByTestId("meta-profile-classifier-input"), {
      target: { value: "minimax" },
    });
    fireEvent.change(screen.getByTestId("meta-profile-default-input"), {
      target: { value: "gpt" },
    });
    fireEvent.change(screen.getByTestId("meta-profile-prompt-template"), {
      target: { value: "Task:\n{{ instance_text }}" },
    });
    fireEvent.change(screen.getByTestId("meta-profile-model-table"), {
      target: { value: "" },
    });
    await user.click(screen.getByTestId("meta-profile-save"));

    await waitFor(() =>
      expect(saveMutateAsync).toHaveBeenCalledWith({
        name: "pareto",
        config: {
          classifier_model: "minimax",
          default_model: "gpt",
          classes: [],
          prompt_template: "Task:\n{{ instance_text }}",
          model_table: null,
        },
      }),
    );
    await waitFor(() =>
      expect(activateMutateAsync).toHaveBeenCalledWith("pareto"),
    );
  });

  it("does not auto-activate a newly-created meta-profile when one is already active", async () => {
    const user = userEvent.setup();
    saveMutateAsync.mockResolvedValue({ name: "pareto" });
    renderWithProviders(<MetaLlmSettingsView />);

    await openMaxScoreTemplate(user);
    await user.clear(screen.getByTestId("meta-profile-name-input"));
    await user.type(screen.getByTestId("meta-profile-name-input"), "pareto");
    fireEvent.change(screen.getByTestId("meta-profile-classifier-input"), {
      target: { value: "minimax" },
    });
    fireEvent.change(screen.getByTestId("meta-profile-default-input"), {
      target: { value: "gpt" },
    });
    fireEvent.change(screen.getByTestId("meta-profile-prompt-template"), {
      target: { value: "Task:\n{{ instance_text }}" },
    });
    fireEvent.change(screen.getByTestId("meta-profile-model-table"), {
      target: { value: "" },
    });
    await user.click(screen.getByTestId("meta-profile-save"));

    await waitFor(() => expect(saveMutateAsync).toHaveBeenCalled());
    expect(activateMutateAsync).not.toHaveBeenCalled();
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
        classes: [],
        prompt_template: "Route this task.\n{{ instance_text }}",
        model_table: "- GPT-5.4",
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
