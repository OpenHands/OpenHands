import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AppSettingsScreen from "#/routes/app-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

const activeBackendState = vi.hoisted(() => ({
  id: "local",
  kind: "local" as "local" | "cloud",
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: activeBackendState.id, kind: activeBackendState.kind },
    orgId: null,
  }),
}));

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
    conversation_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
      ...overrides.conversation_settings,
    },
  };
}

function renderAppSettingsScreen() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<AppSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

/** Serves each backend's own Git identity, as two agent servers would. */
function mockSettingsPerBackend() {
  const gitUserNames: Record<string, string> = {
    local: "openhands",
    "qa-second": "QA Second Git",
  };
  vi.spyOn(SettingsService, "getSettings").mockImplementation(async () =>
    buildSettings({ git_user_name: gitUserNames[activeBackendState.id] }),
  );
}

/** Makes `backendId` active the way the backend selector does: in place. */
function switchBackend(
  rerender: (ui: React.ReactElement) => void,
  backendId: string,
) {
  activeBackendState.id = backendId;
  rerender(<AppSettingsScreen />);
}

describe("AppSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    activeBackendState.id = "local";
    activeBackendState.kind = "local";
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("keeps edits after a failed save and lets the user retry successfully", async () => {
    // Arrange: a real save mutation remains pending until the service rejects.
    const user = userEvent.setup();
    let storedSettings = buildSettings({ enable_sound_notifications: false });
    vi.spyOn(SettingsService, "getSettings").mockImplementation(
      async () => storedSettings,
    );
    let rejectSave!: (error: Error) => void;
    const save = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectSave = reject;
          }),
      );
    renderAppSettingsScreen();
    const sound = await screen.findByTestId(
      "enable-sound-notifications-switch",
    );
    const name = screen.getByTestId("git-user-name-input");

    // Act: edit both a switch and an uncontrolled text input, then save.
    await user.click(sound);
    await user.clear(name);
    await user.type(name, "QA Tester");
    await user.click(screen.getByTestId("submit-button"));

    // Assert: inputs remain mounted during the request, with a pending button.
    expect(sound).toBeInTheDocument();
    expect(screen.getByTestId("submit-button")).toHaveTextContent(
      "SETTINGS$SAVING",
    );
    expect(screen.getByTestId("submit-button")).toBeDisabled();
    rejectSave(new Error("Save failed"));
    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith("Save failed"),
    );
    expect(sound).toBeChecked();
    expect(name).toHaveValue("QA Tester");
    expect(screen.getByTestId("submit-button")).toBeEnabled();

    // Act: retry; the service persists the submitted values and refetches them.
    save.mockImplementationOnce(async (settings) => {
      storedSettings = buildSettings(settings);
      return true;
    });
    await user.click(screen.getByTestId("submit-button"));

    // Assert: the retry sends the same edit, then the form is clean.
    await waitFor(() =>
      expect(displaySuccessToast).toHaveBeenCalledWith("SETTINGS$SAVED"),
    );
    expect(save).toHaveBeenLastCalledWith(
      expect.objectContaining({
        enable_sound_notifications: true,
        git_user_name: "QA Tester",
      }),
    );
    expect(screen.getByTestId("submit-button")).toBeDisabled();
    expect(sound).toBeChecked();
  });

  it("renders the OSS application settings form", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        git_user_name: "octocat",
        git_user_email: "octocat@example.com",
      }),
    );

    renderAppSettingsScreen();

    const analyticsSwitch = await screen.findByTestId(
      "enable-analytics-switch",
    );

    expect(analyticsSwitch).toBeInTheDocument();
    expect(screen.getByTestId("git-user-name-input")).toHaveValue("octocat");
    expect(screen.getByTestId("git-user-email-input")).toHaveValue(
      "octocat@example.com",
    );
    expect(
      screen.getByText("SETTINGS$CONVERSATION_TITLES"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", {
        name: "SETTINGS$TITLE_GENERATION_MODEL",
      }),
    ).toHaveValue("SETTINGS$TITLE_GENERATION_AUTOMATIC");
  });

  it("renders the analytics toggle as checked and disabled for cloud backends", async () => {
    activeBackendState.kind = "cloud";
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ user_consents_to_analytics: false }),
    );

    renderAppSettingsScreen();

    const analyticsSwitch = await screen.findByTestId(
      "enable-analytics-switch",
    );
    const submitButton = screen.getByTestId("submit-button");

    expect(analyticsSwitch).toBeChecked();
    expect(analyticsSwitch).toBeDisabled();

    await userEvent.click(analyticsSwitch);
    expect(submitButton).toBeDisabled();
  });

  it("hides the Getting Started checklist switch on a canvas locked to its Cloud host", async () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    activeBackendState.kind = "cloud";
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderAppSettingsScreen();

    expect(
      await screen.findByTestId("enable-sound-notifications-switch"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("show-getting-started-checklist-switch"),
    ).not.toBeInTheDocument();
  });

  it("does not submit analytics consent for cloud backends", async () => {
    activeBackendState.kind = "cloud";
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ user_consents_to_analytics: false }),
    );

    renderAppSettingsScreen();

    const user = userEvent.setup();
    await user.click(
      await screen.findByTestId("enable-sound-notifications-switch"),
    );
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.not.objectContaining({
          user_consents_to_analytics: expect.anything(),
        }),
      );
    });
  });

  it("saves updated git author details in OSS mode", async () => {
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        git_user_name: "octocat",
        git_user_email: "octocat@example.com",
      }),
    );

    renderAppSettingsScreen();

    const user = userEvent.setup();
    const nameInput = await screen.findByTestId("git-user-name-input");

    await user.clear(nameInput);
    await user.type(nameInput, "monalisa");
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          git_user_name: "monalisa",
          git_user_email: "octocat@example.com",
        }),
      );
    });
  });

  it("saves a dedicated title generation profile", async () => {
    vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
      profiles: [
        {
          name: "Titles",
          model: "anthropic/claude-haiku-3-5",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: null,
    });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderAppSettingsScreen();

    const user = userEvent.setup();
    const input = await screen.findByRole("combobox", {
      name: "SETTINGS$TITLE_GENERATION_MODEL",
    });
    await user.click(input);
    await user.click(
      await screen.findByText("SETTINGS$TITLE_GENERATION_PROFILE_OPTION"),
    );
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title_llm_profile: "Titles",
        }),
      );
    });
  });

  it("saves a custom title generation prompt", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        title_generation_prompt: "Use {conversation_content} as the title.",
      }),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderAppSettingsScreen();

    const user = userEvent.setup();
    const promptInput = await screen.findByRole("textbox", {
      name: "SETTINGS$TITLE_GENERATION_PROMPT",
    });
    await user.clear(promptInput);
    await user.click(promptInput);
    await user.paste("Summarize {conversation_content}");
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title_generation_prompt: "Summarize {conversation_content}",
        }),
      );
    });
  });

  it("resets a custom title generation prompt to the default", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ title_generation_prompt: "Do not use an emoji." }),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderAppSettingsScreen();

    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", {
        name: "SETTINGS$TITLE_GENERATION_PROMPT_RESET",
      }),
    );
    expect(
      screen.getByRole("textbox", {
        name: "SETTINGS$TITLE_GENERATION_PROMPT",
      }),
    ).toHaveValue("");
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title_generation_prompt: null }),
      );
    });
  });

  it("shows the active backend's values after switching back to a backend with cached settings", async () => {
    mockSettingsPerBackend();
    const { rerender } = renderAppSettingsScreen();
    expect(await screen.findByTestId("git-user-name-input")).toHaveValue(
      "openhands",
    );
    switchBackend(rerender, "qa-second");
    await waitFor(() =>
      expect(screen.getByTestId("git-user-name-input")).toHaveValue(
        "QA Second Git",
      ),
    );

    switchBackend(rerender, "local");

    await waitFor(() =>
      expect(screen.getByTestId("git-user-name-input")).toHaveValue(
        "openhands",
      ),
    );
  });

  it("drops unsaved edits on a backend switch and saves only the active backend's values", async () => {
    mockSettingsPerBackend();
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const user = userEvent.setup();
    const { rerender } = renderAppSettingsScreen();
    await screen.findByTestId("git-user-name-input");
    switchBackend(rerender, "qa-second");
    await waitFor(() =>
      expect(screen.getByTestId("git-user-name-input")).toHaveValue(
        "QA Second Git",
      ),
    );
    await user.type(screen.getByTestId("git-user-name-input"), " edited");

    switchBackend(rerender, "local");
    await waitFor(() =>
      expect(screen.getByTestId("git-user-name-input")).toHaveValue(
        "openhands",
      ),
    );
    expect(screen.getByTestId("submit-button")).toBeDisabled();
    await user.click(screen.getByTestId("enable-sound-notifications-switch"));
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() =>
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({ git_user_name: "openhands" }),
      ),
    );
  });
});
