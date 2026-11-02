import { act, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { LlmSettingsLocalView } from "#/components/features/settings/llm-profiles/llm-settings-local-view";
import SettingsService from "#/api/settings-service/settings-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import ProviderConnectionsService from "#/api/provider-connections-service/provider-connections-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { useFreeModelsStore } from "#/stores/free-models-store";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

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
});
