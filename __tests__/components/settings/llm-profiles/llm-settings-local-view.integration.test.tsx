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
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  getActiveSelection,
  getRegisteredBackends,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("LLM profile creation with the real embedded form", () => {
  let inheritedStringSerializer: boolean | null | undefined;
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
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

  async function openNewProfile(user: ReturnType<typeof userEvent.setup>) {
    await user.click(await screen.findByTestId("add-llm-profile"));
    await user.type(
      await screen.findByTestId("llm-api-key-input"),
      "draft-key",
    );
    await waitFor(() =>
      expect(screen.getByTestId("save-profile-btn")).toBeEnabled(),
    );
  }

  it.each([
    ["LLMAuthenticationError", "SETTINGS$PROFILE_CHECK_AUTH_ERROR"],
    ["UnknownProviderError", "SETTINGS$PROFILE_CHECK_GENERIC_ERROR"],
  ])(
    "keeps a safe %s failure until the draft changes, then allows retry",
    async (type, reasonKey) => {
      // Arrange: the backend's diagnostic contains data that must never be rendered.
      vi.mocked(ProfilesService.validateProfile).mockResolvedValueOnce({
        valid: false,
        error: {
          type,
          message: '{"secret":"sk-do-not-display","error":"raw diagnostic"}',
        },
      });
      const save = vi
        .spyOn(ProfilesService, "saveProfile")
        .mockResolvedValue({ name: "draft", message: "Saved" });
      const rename = vi.spyOn(ProfilesService, "renameProfile");
      const user = userEvent.setup();
      renderWithProviders(<LlmSettingsLocalView />);
      await openNewProfile(user);

      // Act: submit through the actual embedded form.
      await user.click(screen.getByTestId("save-profile-btn"));

      // Assert: the form owns the durable feedback and retains the editable draft.
      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent("SETTINGS$PROFILE_CHECK_FAILED");
      expect(alert).toHaveTextContent(reasonKey);
      expect(document.body).not.toHaveTextContent("sk-do-not-display");
      expect(document.body).not.toHaveTextContent("raw diagnostic");
      expect(displayErrorToast).not.toHaveBeenCalled();
      expect(save).not.toHaveBeenCalled();
      expect(rename).not.toHaveBeenCalled();
      expect(screen.getByTestId("llm-api-key-input")).toHaveValue("draft-key");
      expect(screen.queryByTestId("set-indicator")).not.toBeInTheDocument();

      await user.type(screen.getByTestId("llm-api-key-input"), "-corrected");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      await user.click(screen.getByTestId("save-profile-btn"));
      await waitFor(() => expect(save).toHaveBeenCalledOnce());
    },
  );

  it.each([
    {
      label: "false after a field edit",
      result: { valid: false },
      change: "field",
    },
    {
      label: "true after a name edit",
      result: { valid: true },
      change: "name",
    },
    { label: "null after a mode switch", result: null, change: "mode" },
    {
      label: "an error after leaving and reopening",
      result: undefined,
      change: "exit",
    },
  ])(
    "discards $label and releases the pending submission",
    async ({ result, change }) => {
      // Arrange: preflight stays pending while the user revises or leaves the draft.
      const preflight =
        deferred<Awaited<ReturnType<typeof ProfilesService.validateProfile>>>();
      vi.mocked(ProfilesService.validateProfile).mockReturnValueOnce(
        preflight.promise,
      );
      const save = vi
        .spyOn(ProfilesService, "saveProfile")
        .mockResolvedValue({ name: "draft", message: "Saved" });
      const rename = vi.spyOn(ProfilesService, "renameProfile");
      const user = userEvent.setup();
      renderWithProviders(<LlmSettingsLocalView />);
      await openNewProfile(user);
      await user.click(screen.getByTestId("save-profile-btn"));
      await waitFor(() =>
        expect(ProfilesService.validateProfile).toHaveBeenCalledOnce(),
      );

      // Act: real inputs synchronously revoke the old preflight's permission to write.
      if (change === "field") {
        await user.type(screen.getByTestId("llm-api-key-input"), "-new");
      } else if (change === "name") {
        await user.type(screen.getByTestId("profile-name-input"), "New");
      } else if (change === "mode") {
        const allTab = screen.getByTestId("sdk-section-all-toggle");
        await user.click(
          allTab.getAttribute("aria-selected") === "true"
            ? screen.getByTestId("sdk-section-basic-toggle")
            : allTab,
        );
      } else {
        await user.click(screen.getByTestId("back-to-profiles"));
        await user.click(await screen.findByTestId("add-llm-profile"));
        await user.type(
          await screen.findByTestId("llm-api-key-input"),
          "new-draft-key",
        );
      }
      expect(screen.getByTestId("save-profile-btn")).toBeDisabled();
      expect(screen.getByTestId("save-profile-btn")).toHaveTextContent(
        "SETTINGS$PROFILE_WAITING_FOR_SUBMISSION",
      );
      await act(async () => {
        if (result === undefined)
          preflight.reject(new Error("stale backend error"));
        else preflight.resolve(result);
      });

      // Assert: no old result reaches this draft; finally permits a fresh submission.
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(displayErrorToast).not.toHaveBeenCalled();
      expect(displaySuccessToast).not.toHaveBeenCalled();
      expect(save).not.toHaveBeenCalled();
      expect(rename).not.toHaveBeenCalled();
      await waitFor(() =>
        expect(screen.getByTestId("save-profile-btn")).toBeEnabled(),
      );
      await user.click(screen.getByTestId("save-profile-btn"));
      await waitFor(() => expect(save).toHaveBeenCalledOnce());
    },
  );

  it("finishes an active profile write and reapply without closing a newer editor", async () => {
    // Arrange: this profile is already active and its persistence takes time.
    const profile = {
      name: "active",
      model: "openai/gpt-4o",
      base_url: null,
      api_key_set: true,
    };
    vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
      profiles: [profile],
      active_profile: profile.name,
    });
    vi.spyOn(ProfilesService, "getProfile").mockResolvedValue({
      name: profile.name,
      config: { model: profile.model, api_key: "encrypted-key" },
      api_key_set: true,
    });
    const write =
      deferred<Awaited<ReturnType<typeof ProfilesService.saveProfile>>>();
    vi.spyOn(ProfilesService, "saveProfile").mockReturnValueOnce(write.promise);
    const activate = vi
      .spyOn(ProfilesService, "activateProfile")
      .mockResolvedValue({
        name: profile.name,
        message: "Activated",
        llm_applied: true,
      });
    const user = userEvent.setup();
    renderWithProviders(<LlmSettingsLocalView />);
    await user.click(await screen.findByTestId("profile-menu-trigger"));
    await user.click(screen.getByTestId("profile-edit"));
    await user.type(await screen.findByTestId("llm-api-key-input"), "-updated");
    await user.click(screen.getByTestId("save-profile-btn"));
    await waitFor(() =>
      expect(ProfilesService.saveProfile).toHaveBeenCalledOnce(),
    );

    // Act: leave after the write has begun and open a fresh draft.
    await user.click(screen.getByTestId("back-to-profiles"));
    await user.click(await screen.findByTestId("add-llm-profile"));
    await user.type(
      await screen.findByTestId("llm-api-key-input"),
      "new-draft-key",
    );
    await act(async () =>
      write.resolve({ name: profile.name, message: "Saved" }),
    );

    // Assert: the existing write chain completes, but its UI result is obsolete.
    await waitFor(() => expect(activate).toHaveBeenCalledWith(profile.name));
    await waitFor(() =>
      expect(screen.getByTestId("save-profile-btn")).toBeEnabled(),
    );
    expect(screen.getByTestId("llm-api-key-input")).toHaveValue(
      "new-draft-key",
    );
    expect(displaySuccessToast).not.toHaveBeenCalled();
  });

  it.each(["backend change", "unmount"])(
    "revokes a pending preflight on %s",
    async (boundary) => {
      // Arrange: use the real backend provider and registry for this boundary.
      const previousBackends = getRegisteredBackends();
      const previousSelection = getActiveSelection();
      const preflight =
        deferred<Awaited<ReturnType<typeof ProfilesService.validateProfile>>>();
      vi.mocked(ProfilesService.validateProfile).mockReturnValueOnce(
        preflight.promise,
      );
      const save = vi.spyOn(ProfilesService, "saveProfile");
      const user = userEvent.setup();
      const { unmount } = renderWithProviders(
        <ActiveBackendProvider>
          <LlmSettingsLocalView />
        </ActiveBackendProvider>,
      );
      try {
        await openNewProfile(user);
        await user.click(screen.getByTestId("save-profile-btn"));
        await waitFor(() =>
          expect(ProfilesService.validateProfile).toHaveBeenCalledOnce(),
        );

        // Act: change backend context or remove the owner before validation returns.
        if (boundary === "unmount") unmount();
        else {
          act(() => {
            setRegisteredBackends([
              ...previousBackends,
              {
                id: "other-backend",
                name: "Other backend",
                host: "http://localhost:9999",
                apiKey: "",
                kind: "local",
              },
            ]);
            setActiveSelection({ backendId: "other-backend" });
          });
        }
        await act(async () => preflight.resolve({ valid: true }));

        // Assert: an accepted old check cannot authorize persistence in the new context.
        expect(save).not.toHaveBeenCalled();
        expect(displayErrorToast).not.toHaveBeenCalled();
        expect(displaySuccessToast).not.toHaveBeenCalled();
      } finally {
        unmount();
        setRegisteredBackends(previousBackends);
        setActiveSelection(previousSelection);
      }
    },
  );

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
