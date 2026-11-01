import { act, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { LlmSettingsLocalView } from "#/components/features/settings/llm-profiles/llm-settings-local-view";
import SettingsService from "#/api/settings-service/settings-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import ProviderConnectionsService from "#/api/provider-connections-service/provider-connections-service.api";
import ConfigService from "#/api/config-service/config-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { useFreeModelsStore } from "#/stores/free-models-store";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import i18n from "i18next";
import translations from "#/i18n/translation.json";

describe("LLM profile creation with the real embedded form", () => {
  let inheritedStringSerializer: boolean | null | undefined;
  beforeEach(() => {
    vi.restoreAllMocks();
    inheritedStringSerializer = null;
    useFreeModelsStore
      .getState()
      .setFlags({ freeModels: new Set(), defaultModel: "openai/gpt-4o" });
    vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
      profiles: [],
      active_profile: null,
    });
    vi.spyOn(ProfilesService, "validateProfile").mockResolvedValue({
      valid: true,
    });
    vi.spyOn(ProviderConnectionsService, "list").mockResolvedValue([]);
    vi.spyOn(SettingsService, "getSettings").mockImplementation(async () => ({
      ...MOCK_DEFAULT_USER_SETTINGS,
      agent_settings_schema: {
        ...MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
        sections:
          MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!.sections.map(
            (section) =>
              section.key === "llm"
                ? {
                    ...section,
                    fields: [
                      ...section.fields,
                      {
                        key: "llm.aws_secret_access_key",
                        label: "AWS secret access key",
                        description: "Optional provider credential",
                        section: "llm",
                        section_label: "LLM",
                        value_type: "string" as const,
                        default: null,
                        secret: true,
                        required: false,
                        depends_on: [],
                        prominence: "minor" as const,
                        choices: [],
                      },
                      {
                        key: "llm.force_string_serializer",
                        label: "Force string serializer",
                        description: "Null selects automatic model detection",
                        section: "llm",
                        section_label: "LLM",
                        value_type: "boolean" as const,
                        default: null,
                        secret: false,
                        required: false,
                        depends_on: [],
                        prominence: "minor" as const,
                        choices: [],
                      },
                    ],
                  }
                : section,
          ),
      },
      agent_settings: {
        ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
        llm: {
          model: "anthropic/claude-sonnet",
          api_key: "another-profile-key",
          base_url: "https://another-profile.example/v1",
          temperature: 0.4,
          aws_secret_access_key: "**********",
          ...(inheritedStringSerializer === undefined
            ? {}
            : { force_string_serializer: inheritedStringSerializer }),
        },
      },
    }));
  });

  // @spec LLP-001 — New profiles preserve options and credential boundaries.
  it("saves an untouched displayed option without inheriting another profile's credentials", async () => {
    // Arrange: existing agent settings have a non-default option and credentials.
    const save = vi
      .spyOn(ProfilesService, "saveProfile")
      .mockResolvedValue({ name: "QA_Inherited", message: "Saved" });
    const user = userEvent.setup();
    renderWithProviders(<LlmSettingsLocalView />);

    // Act: create a profile, keeping its displayed temperature unchanged.
    await user.click(await screen.findByTestId("add-llm-profile"));
    await user.click(await screen.findByTestId("sdk-section-all-toggle"));
    expect(screen.getByTestId("sdk-settings-llm.temperature")).toHaveValue(0.4);
    expect(screen.getByTestId("llm-api-key-input")).toHaveValue("");
    expect(screen.getByTestId("base-url-input")).toHaveValue("");
    expect(
      screen.getByTestId("sdk-settings-llm.aws_secret_access_key"),
    ).toHaveValue("");
    await user.clear(screen.getByTestId("profile-name-input"));
    await user.type(screen.getByTestId("profile-name-input"), "QA_Inherited");
    await user.type(screen.getByTestId("llm-api-key-input"), "new-profile-key");
    await user.click(screen.getByTestId("save-profile-btn"));

    // Assert: the real schema form's coerced option reaches profile persistence.
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith("QA_Inherited", {
        llm: expect.objectContaining({
          model: "openai/gpt-4o",
          api_key: "new-profile-key",
          temperature: 0.4,
        }),
        include_secrets: true,
      }),
    );
    expect(save.mock.calls[0][1].llm.base_url).not.toBe(
      "https://another-profile.example/v1",
    );
    expect(save.mock.calls[0][1].llm.aws_secret_access_key).toBeUndefined();
  });

  // @spec LLP-001 — Preserve automatic nullable booleans and explicit overrides.
  it.each([
    {
      name: "untouched automatic null",
      initial: null,
      edit: false,
      refresh: undefined,
      expected: undefined,
    },
    {
      name: "untouched omitted automatic value",
      initial: undefined,
      edit: false,
      refresh: undefined,
      expected: undefined,
    },
    {
      name: "inherited explicit false",
      initial: false,
      edit: false,
      refresh: undefined,
      expected: false,
    },
    {
      name: "explicit dirty edit",
      initial: null,
      edit: true,
      refresh: undefined,
      expected: true,
    },
    {
      name: "inherited false after settings refresh",
      initial: false,
      edit: false,
      refresh: null,
      expected: false,
    },
  ])(
    "keeps $name in the new profile",
    async ({ initial, edit, refresh, expected }) => {
      // Arrange: null and false render alike but have different provider meaning.
      inheritedStringSerializer = initial;
      const save = vi
        .spyOn(ProfilesService, "saveProfile")
        .mockResolvedValue({ name: "QA_Boolean", message: "Saved" });
      const user = userEvent.setup();
      const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      renderWithProviders(
        <QueryClientProvider client={queryClient}>
          <LlmSettingsLocalView />
        </QueryClientProvider>,
      );
      await user.click(await screen.findByTestId("add-llm-profile"));
      await user.click(await screen.findByTestId("sdk-section-all-toggle"));
      const checkbox = screen.getByTestId(
        "sdk-settings-llm.force_string_serializer",
      );
      expect(checkbox).not.toBeChecked();

      // Act: save untouched, or explicitly edit, without re-seeding the open draft.
      if (refresh !== undefined) {
        inheritedStringSerializer = refresh;
        await act(async () => {
          await queryClient.invalidateQueries({
            queryKey: SETTINGS_QUERY_KEYS.byScope("personal"),
          });
        });
      }
      if (edit) await user.click(checkbox.closest("label")!);
      await user.clear(screen.getByTestId("profile-name-input"));
      await user.type(screen.getByTestId("profile-name-input"), "QA_Boolean");
      await user.type(
        screen.getByTestId("llm-api-key-input"),
        "new-profile-key",
      );
      await user.click(screen.getByTestId("save-profile-btn"));

      // Assert: the saved request preserves auto-detection, false or the dirty true.
      await waitFor(() => expect(save).toHaveBeenCalled());
      expect(save.mock.calls[0][1].llm.force_string_serializer).toBe(expected);
    },
  );

  it("opens a prefilled custom-model profile from the Add models hint", async () => {
    // Arrange: an OpenRouter connection whose catalog lacks the wanted model.
    vi.spyOn(ProviderConnectionsService, "list").mockResolvedValue([
      {
        id: "conn-openrouter",
        display_name: "OpenRouter",
        provider: "openrouter",
        base_url: null,
        created_at: 1,
        updated_at: 1,
        api_key_set: true,
      },
    ]);
    vi.spyOn(ConfigService, "searchProviders").mockResolvedValue({
      items: [{ name: "openrouter", verified: true }],
      next_page_id: null,
    });
    vi.spyOn(ConfigService, "searchModels").mockResolvedValue({
      items: [],
      next_page_id: null,
    });
    const save = vi
      .spyOn(ProfilesService, "saveProfile")
      .mockResolvedValue({ name: "nemotron", message: "Saved" });
    const user = userEvent.setup();
    renderWithProviders(<LlmSettingsLocalView />);

    // The test i18n has no strings, and the link only exists inside the hint's
    // translated sentence, so load the real English hint.
    i18n.addResourceBundle(
      "en",
      "openhands",
      {
        SETTINGS$ADD_MODELS_CUSTOM_MODEL_HINT:
          translations.SETTINGS$ADD_MODELS_CUSTOM_MODEL_HINT.en,
      },
      true,
      true,
    );

    // Act: follow the hint's Add LLM Profile link from the connection's modal.
    await user.click(
      await screen.findByTestId("provider-connection-group-add-models"),
    );
    // The link must carry the label itself, not render empty beside it.
    const link = await screen.findByTestId("add-models-custom-model-link");
    expect(link).toHaveTextContent("SETTINGS$ADD_LLM_PROFILE");
    await user.click(link);

    // Assert: the form opens on Advanced, linked, with only the provider prefix.
    expect(
      await screen.findByTestId("sdk-section-advanced-toggle"),
    ).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("llm-custom-model-input")).toHaveValue(
      "openrouter/",
    );
    expect(screen.queryByTestId("llm-api-key-input")).not.toBeInTheDocument();

    // A bare prefix names no model, so saving it is refused.
    await user.click(screen.getByTestId("save-profile-btn"));
    expect(save).not.toHaveBeenCalled();

    await user.type(
      screen.getByTestId("llm-custom-model-input"),
      "nvidia/nemotron-3.5-super-vl:free",
    );
    await user.click(screen.getByTestId("save-profile-btn"));
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          llm: expect.objectContaining({
            model: "openrouter/nvidia/nemotron-3.5-super-vl:free",
            provider_connection_id: "conn-openrouter",
          }),
        }),
      ),
    );
  });
});
