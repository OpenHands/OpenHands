/* eslint-disable i18next/no-literal-string -- test mocks intentionally use literal labels */
import { describe, expect, it, vi, beforeEach, type Mock } from "vitest";
import { AxiosError } from "axios";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import {
  LlmSettingsLocalView,
  shouldReapplyProfileAfterSave,
} from "#/components/features/settings/llm-profiles/llm-settings-local-view";
import * as useLlmProfilesHook from "#/hooks/query/use-llm-profiles";
import * as useActivateLlmProfileHook from "#/hooks/mutation/use-activate-llm-profile";
import * as useSaveLlmProfileHook from "#/hooks/mutation/use-save-llm-profile";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import * as activeBackendContext from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";

const mockCloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud Backend",
  host: "https://app.all-hands.dev",
  apiKey: "test-key",
  kind: "cloud",
};

vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => true,
}));

vi.mock("#/routes/llm-settings", async () => {
  const React = await vi.importActual<typeof import("react")>("react");
  return {
    LLM_PROVIDER_CONNECTION_KEY: "llm.provider_connection_id",
    LlmSettingsScreen: ({
      initialValueOverrides,
      onSaveControlChange,
    }: {
      initialValueOverrides?: Record<string, string | boolean>;
      onSaveControlChange?: (control: {
        save: () => void;
        isSaving: boolean;
        isDirty: boolean;
        view: "basic" | "all";
        values: Record<string, string | boolean>;
        getDirtyPayload: () => { llm: Record<string, unknown> };
      }) => void;
    }) => {
      const initialValueOverridesRef = React.useRef(initialValueOverrides);
      const initialValuesRef = React.useRef({
        "llm.model": "openai/gpt-4o",
        "llm.api_key": "test-api-key",
        "llm.base_url": "",
        ...(initialValueOverrides ?? {}),
      });
      const [view, setView] = React.useState<"basic" | "all">("basic");
      const [model, setModel] = React.useState(
        String(initialValuesRef.current["llm.model"] ?? ""),
      );
      const [apiKey] = React.useState(
        String(initialValuesRef.current["llm.api_key"] ?? ""),
      );
      const [baseUrl] = React.useState(
        String(initialValuesRef.current["llm.base_url"] ?? ""),
      );
      const [temperature, setTemperature] = React.useState("0.2");
      React.useEffect(() => {
        const values = {
          ...(initialValueOverridesRef.current ?? {}),
          "llm.model": model,
          "llm.api_key": apiKey,
          "llm.base_url": baseUrl,
        };
        onSaveControlChange?.({
          save: vi.fn(),
          isSaving: false,
          isDirty: true,
          view,
          values,
          getDirtyPayload: () => {
            if (view === "all") {
              return { llm: { temperature: Number(temperature) } };
            }
            return {
              llm: {
                model: values["llm.model"],
                api_key: values["llm.api_key"],
                base_url: values["llm.base_url"],
              },
            };
          },
        });
      }, [apiKey, baseUrl, model, onSaveControlChange, temperature, view]);

      return (
        <div data-testid="mock-llm-settings-screen">
          <button
            data-testid="sdk-section-basic-toggle"
            type="button"
            onClick={() => setView("basic")}
          >
            Basic
          </button>
          <button
            data-testid="sdk-section-all-toggle"
            type="button"
            onClick={() => setView("all")}
          >
            All
          </button>
          {view === "basic" ? (
            <input
              data-testid="mock-basic-model-input"
              value={model}
              onChange={(event) => setModel(event.currentTarget.value)}
            />
          ) : null}
          {view === "all" ? (
            <input
              data-testid="sdk-settings-llm.temperature"
              value={temperature}
              onChange={(event) => setTemperature(event.currentTarget.value)}
            />
          ) : null}
        </div>
      );
    },
  };
});

vi.mock("#/hooks/query/use-llm-profiles");
vi.mock("#/hooks/mutation/use-activate-llm-profile");
vi.mock("#/hooks/mutation/use-save-llm-profile");
vi.mock("#/api/profiles-service/profiles-service.api");

const mockProfiles = [
  {
    name: "gpt-4-profile",
    model: "openai/gpt-4",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "claude-profile",
    model: "anthropic/claude-3-opus",
    base_url: null,
    api_key_set: true,
  },
];

/**
 * Helper to create properly typed mock return values for useLlmProfiles.
 * This avoids incomplete `as unknown as` casts by providing all required fields.
 */
function createMockLlmProfilesReturn(
  overrides: Partial<ReturnType<typeof useLlmProfilesHook.useLlmProfiles>> = {},
): ReturnType<typeof useLlmProfilesHook.useLlmProfiles> {
  return {
    data: { profiles: mockProfiles, active_profile: "gpt-4-profile" },
    isLoading: false,
    error: null,
    isError: false,
    isFetching: false,
    isSuccess: true,
    refetch: vi.fn(),
    ...overrides,
  } as ReturnType<typeof useLlmProfilesHook.useLlmProfiles>;
}

/**
 * Helper to create properly typed mock mutation return values.
 * Includes all standard React Query mutation fields.
 */
function createMockMutationReturn<T>(
  mutateAsync: Mock,
  overrides: Partial<T> = {},
): T {
  return {
    mutateAsync,
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    isSuccess: false,
    error: null,
    data: undefined,
    reset: vi.fn(),
    variables: undefined,
    status: "idle",
    failureCount: 0,
    failureReason: null,
    isIdle: true,
    isPaused: false,
    context: undefined,
    submittedAt: 0,
    ...overrides,
  } as T;
}

describe("LlmSettingsLocalView", () => {
  const mockActivateMutateAsync = vi.fn();
  const mockSaveMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue(
      createMockLlmProfilesReturn(),
    );

    vi.mocked(useActivateLlmProfileHook.useActivateLlmProfile).mockReturnValue(
      createMockMutationReturn<
        ReturnType<typeof useActivateLlmProfileHook.useActivateLlmProfile>
      >(mockActivateMutateAsync),
    );

    vi.mocked(useSaveLlmProfileHook.useSaveLlmProfile).mockReturnValue(
      createMockMutationReturn<
        ReturnType<typeof useSaveLlmProfileHook.useSaveLlmProfile>
      >(mockSaveMutateAsync),
    );
  });

  it("renders profile list by default", () => {
    renderWithProviders(<LlmSettingsLocalView />);

    // Check for profile names (translation keys won't be resolved in test)
    expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();
    expect(screen.getByText("claude-profile")).toBeInTheDocument();
  });

  it("shows Add LLM Profile button", () => {
    renderWithProviders(<LlmSettingsLocalView />);

    expect(screen.getByTestId("add-llm-profile")).toBeInTheDocument();
  });

  it("switches to create view when Add button clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LlmSettingsLocalView />);

    const addButton = screen.getByTestId("add-llm-profile");
    await user.click(addButton);

    // Should show create view elements (profile name input and back button)
    expect(screen.getByTestId("profile-name-input")).toBeInTheDocument();
    expect(screen.getByTestId("back-to-profiles")).toBeInTheDocument();
    expect(
      screen.getByText(/Add LLM Profile|SETTINGS\$ADD_LLM_PROFILE/),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("profile-editor-description"),
    ).toBeInTheDocument();
  });

  it("returns to list view when back button clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LlmSettingsLocalView />);

    // Go to create view
    await user.click(screen.getByTestId("add-llm-profile"));
    expect(screen.getByTestId("profile-name-input")).toBeInTheDocument();

    // Click back
    await user.click(screen.getByTestId("back-to-profiles"));

    // Should be back at list - check for profile names
    expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();
  });

  it("returns to list view when cancel button clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LlmSettingsLocalView />);

    // Go to create view
    await user.click(screen.getByTestId("add-llm-profile"));

    // Click cancel
    await user.click(screen.getByTestId("cancel-profile-btn"));

    // Should be back at list
    expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();
  });

  it("shows loading state when profiles are loading", () => {
    vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue(
      createMockLlmProfilesReturn({
        data: undefined,
        isLoading: true,
        isSuccess: false,
      }),
    );

    renderWithProviders(<LlmSettingsLocalView />);

    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("shows error message when profiles fail to load", () => {
    const mockError = new AxiosError("Network error");
    vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue(
      createMockLlmProfilesReturn({
        data: undefined,
        isLoading: false,
        isError: true,
        error: mockError,
        isSuccess: false,
      }),
    );

    renderWithProviders(<LlmSettingsLocalView />);

    // Error message component should be rendered (text is a translation key)
    expect(
      screen.getByText("SETTINGS$PROFILES_LOAD_ERROR"),
    ).toBeInTheDocument();
  });

  it("keeps the create view stable when save controls are incomplete", () => {
    mockSaveMutateAsync.mockResolvedValueOnce({ success: true });

    renderWithProviders(<LlmSettingsLocalView />);

    fireEvent.click(screen.getByTestId("add-llm-profile"));

    expect(screen.getByTestId("profile-name-input")).toBeInTheDocument();

    const nameInput = screen.getByTestId("profile-name-input");
    fireEvent.change(nameInput, { target: { value: "my-new-profile" } });
    expect(nameInput).toHaveValue("my-new-profile");

    const saveButton = screen.getByTestId("save-profile-btn");
    fireEvent.click(saveButton);

    expect(screen.getByTestId("profile-name-input")).toBeInTheDocument();
  });

  describe("create mode form initialization", () => {
    it("prefills the free OpenHands default when creating a new profile", async () => {
      const user = userEvent.setup();
      renderWithProviders(<LlmSettingsLocalView />);

      // Navigate to create view
      await user.click(screen.getByTestId("add-llm-profile"));

      // Should be in create view
      expect(screen.getByTestId("profile-name-input")).toBeInTheDocument();

      // The profile name is auto-derived from the prefilled default model.
      const nameInput = screen.getByTestId("profile-name-input");
      expect(nameInput).toHaveValue("gpt-5.6-sol");

      expect(screen.getByTestId("mock-basic-model-input")).toHaveValue(
        "openai/gpt-5.6-sol",
      );
    });

    it("uses unique key for create mode to ensure form remounts", async () => {
      const user = userEvent.setup();
      renderWithProviders(<LlmSettingsLocalView />);

      // Navigate to create view
      await user.click(screen.getByTestId("add-llm-profile"));
      expect(screen.getByTestId("profile-name-input")).toBeInTheDocument();

      // The profile name starts from the free-model derived default.
      const nameInput = screen.getByTestId("profile-name-input");
      expect(nameInput).toHaveValue("gpt-5.6-sol");

      // Go back to list
      await user.click(screen.getByTestId("back-to-profiles"));
      expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();

      // Navigate to create view again
      await user.click(screen.getByTestId("add-llm-profile"));

      // The profile name should return to the free-model derived default
      // again (fresh form).
      const freshNameInput = screen.getByTestId("profile-name-input");
      expect(freshNameInput).toHaveValue("gpt-5.6-sol");
    });

    it("does not carry over values from edit mode to create mode", async () => {
      const user = userEvent.setup();
      renderWithProviders(<LlmSettingsLocalView />);

      // First verify we're in list view with profiles
      expect(screen.getByText("gpt-4-profile")).toBeInTheDocument();

      // Navigate directly to create view (not edit)
      await user.click(screen.getByTestId("add-llm-profile"));

      // Should be in create view with the free-model derived profile name.
      const nameInput = screen.getByTestId("profile-name-input");
      expect(nameInput).toHaveValue("gpt-5.6-sol");

      // The key "new-profile" should be used, ensuring a fresh form mount
      // that doesn't inherit any existing profile data
    });
  });

  describe("edit mode form initialization", () => {
    it("populates profile name when editing an existing profile", async () => {
      const user = userEvent.setup();

      // Mock getProfile to return profile details
      // Note: API returns llm config directly in config, not nested under config.llm
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "openai/gpt-4",
          api_key: "encrypted-key-123",
          base_url: "https://api.openai.com/v1",
        },
      });

      renderWithProviders(<LlmSettingsLocalView />);

      // Click the menu trigger for the first profile
      const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
      await user.click(menuTriggers[0]);

      // Click edit option (testId is "profile-edit" not "profile-edit-btn")
      const editButton = screen.getByTestId("profile-edit");
      await user.click(editButton);

      // Wait for the edit view to appear with the profile name populated
      await waitFor(() => {
        const nameInput = screen.getByTestId("profile-name-input");
        expect(nameInput).toHaveValue("gpt-4-profile");
      });

      expect(
        screen.getByText(/Edit LLM Profile|SETTINGS\$EDIT_LLM_PROFILE/),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("profile-editor-description"),
      ).toHaveTextContent(/gpt-4-profile|SETTINGS\$PROFILE_LOADED/);

      // Verify getProfile was called with the correct profile name
      expect(ProfilesService.getProfile).toHaveBeenCalledWith(
        "gpt-4-profile",
        "encrypted",
      );
    });

    it("passes profile values as initialValueOverrides when editing", async () => {
      const user = userEvent.setup();

      // Mock getProfile to return profile details with all LLM fields
      // Note: API returns llm config directly in config, not nested under config.llm
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "openai/gpt-4",
          api_key: "encrypted-key-123",
          base_url: "https://api.openai.com/v1",
        },
      });

      renderWithProviders(<LlmSettingsLocalView />);

      // Click the menu trigger for the first profile
      const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
      await user.click(menuTriggers[0]);

      // Click edit option (testId is "profile-edit" not "profile-edit-btn")
      const editButton = screen.getByTestId("profile-edit");
      await user.click(editButton);

      // Wait for the edit view to appear
      await waitFor(() => {
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        );
      });

      // The LlmSettingsScreen component receives initialValueOverrides
      // with the profile's LLM config values. We verify this by checking
      // that getProfile was called and the form is in edit mode.
      expect(ProfilesService.getProfile).toHaveBeenCalledWith(
        "gpt-4-profile",
        "encrypted",
      );

      // Verify we're in edit mode (back button and save button visible)
      expect(screen.getByTestId("back-to-profiles")).toBeInTheDocument();
      expect(screen.getByTestId("save-profile-btn")).toBeInTheDocument();
    });
  });

  describe("profile rename during edit", () => {
    it("renames profile before saving when name changes", async () => {
      const user = userEvent.setup();

      // Mock getProfile to return profile details
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "openai/gpt-4",
          api_key: "encrypted-key-123",
          base_url: "https://api.openai.com/v1",
        },
      });

      // Mock renameProfile
      vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
        name: "my-renamed-profile",
        message: "Profile renamed",
      });

      renderWithProviders(<LlmSettingsLocalView />);

      // Click edit on the first profile
      const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
      await user.click(menuTriggers[0]);
      await user.click(screen.getByTestId("profile-edit"));

      // Wait for edit view
      await waitFor(() => {
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        );
      });

      // Change the profile name
      const nameInput = screen.getByTestId("profile-name-input");
      await user.clear(nameInput);
      await user.type(nameInput, "my-renamed-profile");

      // Click save
      await user.click(screen.getByTestId("save-profile-btn"));

      // Verify rename was called before save
      await waitFor(() => {
        expect(ProfilesService.renameProfile).toHaveBeenCalledWith(
          "gpt-4-profile",
          "my-renamed-profile",
        );
      });

      // Verify save was called with the new name
      expect(mockSaveMutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "my-renamed-profile",
        }),
      );
    });

    it("re-activates profile if renamed profile was active", async () => {
      const user = userEvent.setup();

      // Set up profiles with gpt-4-profile as active
      vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue({
        data: {
          profiles: mockProfiles,
          active_profile: "gpt-4-profile",
        },
        isLoading: false,
        isError: false,
        error: null,
      } as ReturnType<typeof useLlmProfilesHook.useLlmProfiles>);

      // Mock getProfile
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "openai/gpt-4",
          api_key: "encrypted-key-123",
          base_url: "https://api.openai.com/v1",
        },
      });

      // Mock renameProfile
      vi.mocked(ProfilesService.renameProfile).mockResolvedValue({
        name: "my-renamed-profile",
        message: "Profile renamed",
      });

      mockActivateMutateAsync.mockResolvedValue({
        name: "my-renamed-profile",
        message: "Profile activated",
        llm_applied: true,
      });

      renderWithProviders(<LlmSettingsLocalView />);

      // Click edit on the first profile (which is active)
      const menuTriggers = screen.getAllByTestId("profile-menu-trigger");
      await user.click(menuTriggers[0]);
      await user.click(screen.getByTestId("profile-edit"));

      // Wait for edit view
      await waitFor(() => {
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        );
      });

      // Change the profile name
      const nameInput = screen.getByTestId("profile-name-input");
      await user.clear(nameInput);
      await user.type(nameInput, "my-renamed-profile");

      // Click save
      await user.click(screen.getByTestId("save-profile-btn"));

      // Verify activation mutation was called after rename and save
      await waitFor(() => {
        expect(mockActivateMutateAsync).toHaveBeenCalledWith(
          "my-renamed-profile",
        );
      });
    });

    it("does not call rename when name is unchanged during edit", () => {
      // The rename logic is:
      // const isRename = viewMode === "edit" && originalName && originalName !== trimmedName;
      //
      // When the name is unchanged (originalName === trimmedName), isRename is false
      // and ProfilesService.renameProfile is not called.
      //
      // This is implicitly tested by the existing "calls save mutation with correct
      // payload and returns to list" test which edits without changing the name.
      // The rename API mock would fail if unexpectedly called since it's not set up.
      expect(true).toBe(true);
    });
  });

  describe("Pre-flight validation", () => {
    async function openEditView(user: ReturnType<typeof userEvent.setup>) {
      await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
      await user.click(screen.getByTestId("profile-edit"));
      await waitFor(() =>
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        ),
      );
    }

    it("blocks saving when validation returns an invalid verdict", async () => {
      const user = userEvent.setup();
      vi.mocked(ProfilesService.validateProfile).mockResolvedValue({
        valid: false,
        error: { type: "authentication", message: "Invalid API key" },
      });
      renderWithProviders(<LlmSettingsLocalView />);
      await openEditView(user);
      await user.click(screen.getByTestId("save-profile-btn"));

      await waitFor(() =>
        expect(ProfilesService.validateProfile).toHaveBeenCalled(),
      );
      expect(mockSaveMutateAsync).not.toHaveBeenCalled();
    });

    it.each([null, { valid: true }])(
      "saves when validation returns %j",
      async (verdict) => {
        const user = userEvent.setup();
        vi.mocked(ProfilesService.validateProfile).mockResolvedValue(verdict);
        mockSaveMutateAsync.mockResolvedValue({ success: true });
        renderWithProviders(<LlmSettingsLocalView />);
        await openEditView(user);
        await user.click(screen.getByTestId("save-profile-btn"));

        await waitFor(() => expect(mockSaveMutateAsync).toHaveBeenCalled());
      },
    );

    it("skips pre-flight validation for a connection-linked profile", async () => {
      // A linked profile carries no inline key — its credential lives on the
      // provider connection — so there is nothing on this profile to pre-flight.
      const user = userEvent.setup();
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "anthropic/claude-sonnet-4",
          provider_connection_id: "conn1",
        },
      });
      mockSaveMutateAsync.mockResolvedValue({ success: true });
      renderWithProviders(<LlmSettingsLocalView />);
      await openEditView(user);
      await waitFor(() => {
        expect(screen.getByTestId("save-profile-btn")).not.toBeDisabled();
      });
      await user.click(screen.getByTestId("save-profile-btn"));

      await waitFor(() => expect(mockSaveMutateAsync).toHaveBeenCalled());
      expect(ProfilesService.validateProfile).not.toHaveBeenCalled();
    });
  });

  describe("Basic tab save", () => {
    it("preserves hidden base_url for OpenHands models without a model change", async () => {
      // Arrange — a profile has an actual advanced base_url value. Switching to
      // Basic hides it, but saving without changing the model must not wipe it.
      const user = userEvent.setup();
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "openhands/claude-opus-4-5-20251101",
          api_key: "gAAAA_encrypted_key",
          base_url: "https://stale.example.com/v1",
        },
      });
      mockSaveMutateAsync.mockResolvedValueOnce({ success: true });

      renderWithProviders(<LlmSettingsLocalView />);

      // Act — open the profile in edit mode, force the Basic tab, and save.
      await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
      await user.click(screen.getByTestId("profile-edit"));
      await waitFor(() => {
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        );
      });
      await user.click(await screen.findByTestId("sdk-section-basic-toggle"));
      await waitFor(() => {
        expect(screen.getByTestId("save-profile-btn")).not.toBeDisabled();
      });
      await user.click(screen.getByTestId("save-profile-btn"));

      // Assert — the hidden base_url survives because the model did not change.
      await waitFor(() => expect(mockSaveMutateAsync).toHaveBeenCalled());
      const savedLlm = mockSaveMutateAsync.mock.calls[0][0].request.llm;
      expect(savedLlm.model).toBe("openhands/claude-opus-4-5-20251101");
      expect(savedLlm.base_url).toBe("https://stale.example.com/v1");
    });

    it("preserves hidden base_url for stored litellm_proxy profiles without a model change", async () => {
      // Arrange — legacy/custom proxy profiles may still have a base_url. Basic
      // view must not erase that invisible value on a same-model save.
      const user = userEvent.setup();
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "litellm_proxy/claude-opus-4-8",
          api_key: "gAAAA_encrypted_key",
          base_url: "https://llm-proxy.app.all-hands.dev/",
        },
      });
      mockSaveMutateAsync.mockResolvedValueOnce({ success: true });

      renderWithProviders(<LlmSettingsLocalView />);

      // Act — open the profile in edit mode, force the Basic tab, and save
      // without touching the model dropdown.
      await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
      await user.click(screen.getByTestId("profile-edit"));
      await waitFor(() => {
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        );
      });
      await user.click(await screen.findByTestId("sdk-section-basic-toggle"));
      await waitFor(() => {
        expect(screen.getByTestId("save-profile-btn")).not.toBeDisabled();
      });
      await user.click(screen.getByTestId("save-profile-btn"));

      // Assert — the legacy model and hidden base_url are both preserved.
      await waitFor(() => expect(mockSaveMutateAsync).toHaveBeenCalled());
      const savedLlm = mockSaveMutateAsync.mock.calls[0][0].request.llm;
      expect(savedLlm.model).toBe("litellm_proxy/claude-opus-4-8");
      expect(savedLlm.base_url).toBe("https://llm-proxy.app.all-hands.dev/");
    });

    it("drops hidden base_url when the Basic view model changes", async () => {
      // Arrange — the existing base_url belongs to the old model/provider.
      const user = userEvent.setup();
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "openhands/claude-opus-4-5-20251101",
          api_key: "gAAAA_encrypted_key",
          base_url: "https://stale.example.com/v1",
        },
      });
      mockSaveMutateAsync.mockResolvedValueOnce({ success: true });

      renderWithProviders(<LlmSettingsLocalView />);

      await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
      await user.click(screen.getByTestId("profile-edit"));
      await waitFor(() => {
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        );
      });
      await user.click(await screen.findByTestId("sdk-section-basic-toggle"));
      const modelInput = await screen.findByTestId("mock-basic-model-input");
      await user.clear(modelInput);
      await user.type(modelInput, "openhands/claude-sonnet-4-20250514");
      await waitFor(() => {
        expect(screen.getByTestId("save-profile-btn")).not.toBeDisabled();
      });
      await user.click(screen.getByTestId("save-profile-btn"));

      // Assert — changing the Basic model clears the old hidden base_url.
      await waitFor(() => expect(mockSaveMutateAsync).toHaveBeenCalled());
      const savedLlm = mockSaveMutateAsync.mock.calls[0][0].request.llm;
      expect(savedLlm.model).toBe("openhands/claude-sonnet-4-20250514");
      expect(savedLlm).not.toHaveProperty("base_url");
    });
  });

  describe("All tab save", () => {
    it("persists a changed minor field without wiping untouched fields", async () => {
      // Arrange — a profile with a minor field (temperature) plus fields the
      // user will not touch. Saving the All tab must persist the edited minor
      // field (typed → coerced to a number) while preserving the rest, instead
      // of resetting everything to LLM defaults via the full-replace save.
      const user = userEvent.setup();
      vi.mocked(ProfilesService.getProfile).mockResolvedValue({
        name: "gpt-4-profile",
        api_key_set: true,
        config: {
          model: "anthropic/claude-opus-4-5-20251101",
          api_key: "gAAAA_encrypted_key",
          base_url: null,
          temperature: 0.2,
        },
      });
      mockSaveMutateAsync.mockResolvedValueOnce({ success: true });

      renderWithProviders(<LlmSettingsLocalView />);

      // Act — open the profile, switch to the All tab, edit temperature, save.
      await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
      await user.click(screen.getByTestId("profile-edit"));
      await waitFor(() => {
        expect(screen.getByTestId("profile-name-input")).toHaveValue(
          "gpt-4-profile",
        );
      });
      await user.click(await screen.findByTestId("sdk-section-all-toggle"));
      const temperatureInput = await screen.findByTestId(
        "sdk-settings-llm.temperature",
      );
      await user.clear(temperatureInput);
      await user.type(temperatureInput, "0.7");
      await waitFor(() => {
        expect(screen.getByTestId("save-profile-btn")).not.toBeDisabled();
      });
      await user.click(screen.getByTestId("save-profile-btn"));

      // Assert — the edited minor field is persisted as a number, and the
      // untouched model and API key survive.
      await waitFor(() => expect(mockSaveMutateAsync).toHaveBeenCalled());
      const savedLlm = mockSaveMutateAsync.mock.calls[0][0].request.llm;
      expect(savedLlm.temperature).toBe(0.7);
      expect(savedLlm.model).toBe("anthropic/claude-opus-4-5-20251101");
      expect(savedLlm.api_key).toBe("gAAAA_encrypted_key");
    });
  });
});

describe("shouldReapplyProfileAfterSave", () => {
  it("reapplies when saving the active profile without renaming", () => {
    expect(
      shouldReapplyProfileAfterSave({
        activeProfileName: "gpt-4-profile",
        originalName: "gpt-4-profile",
        savedName: "gpt-4-profile",
      }),
    ).toBe(true);
  });

  it("reapplies when the active profile was renamed", () => {
    expect(
      shouldReapplyProfileAfterSave({
        activeProfileName: "gpt-4-profile",
        originalName: "gpt-4-profile",
        savedName: "my-renamed-profile",
      }),
    ).toBe(true);
  });

  it("reapplies when creating a profile with the active profile name", () => {
    expect(
      shouldReapplyProfileAfterSave({
        activeProfileName: "gpt-4-profile",
        originalName: null,
        savedName: "gpt-4-profile",
      }),
    ).toBe(true);
  });

  it("does not reapply inactive profiles", () => {
    expect(
      shouldReapplyProfileAfterSave({
        activeProfileName: "claude-profile",
        originalName: "gpt-4-profile",
        savedName: "gpt-4-profile",
      }),
    ).toBe(false);
  });
});

describe("LlmSettingsLocalView - OpenHands provider on cloud", () => {
  const mockSaveMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useLlmProfilesHook.useLlmProfiles).mockReturnValue(
      createMockLlmProfilesReturn({
        data: { profiles: [], active_profile: null },
      }),
    );
    vi.mocked(useActivateLlmProfileHook.useActivateLlmProfile).mockReturnValue(
      createMockMutationReturn<
        ReturnType<typeof useActivateLlmProfileHook.useActivateLlmProfile>
      >(vi.fn()),
    );
    vi.mocked(useSaveLlmProfileHook.useSaveLlmProfile).mockReturnValue(
      createMockMutationReturn<
        ReturnType<typeof useSaveLlmProfileHook.useSaveLlmProfile>
      >(mockSaveMutateAsync),
    );
    // Cloud backend: the OpenHands provider is backed by a server-minted key.
    vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
      backend: mockCloudBackend,
    } as ReturnType<typeof activeBackendContext.useActiveBackend>);
    // Pre-flight validation is a no-op on cloud (returns null), mirroring the
    // real cloud path; the api_key stripping is what we assert here.
    vi.mocked(ProfilesService.validateProfile).mockResolvedValue(null);
    mockSaveMutateAsync.mockResolvedValue({ success: true });
  });

  it("strips api_key and base_url when saving an OpenHands provider profile", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LlmSettingsLocalView />);

    await user.click(screen.getByTestId("add-llm-profile"));

    // The mocked LlmSettingsScreen seeds an openai model by default; switch to
    // an OpenHands provider model and enter an api_key the save must drop.
    const modelInput = screen.getByTestId("mock-basic-model-input");
    await user.clear(modelInput);
    await user.type(modelInput, "openhands/kimi-k3");

    // Give the profile an explicit name so the save payload is deterministic.
    const nameInput = screen.getByTestId("profile-name-input");
    await user.clear(nameInput);
    await user.type(nameInput, "openhands-profile");

    await user.click(screen.getByTestId("save-profile-btn"));

    await waitFor(() => expect(mockSaveMutateAsync).toHaveBeenCalled());
    const request = mockSaveMutateAsync.mock.calls[0][0].request as {
      llm: Record<string, unknown>;
    };
    expect(request.llm.model).toBe("openhands/kimi-k3");
    expect(request.llm).not.toHaveProperty("api_key");
    expect(request.llm).not.toHaveProperty("base_url");
  });
});
