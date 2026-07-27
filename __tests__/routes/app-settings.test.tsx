import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AppSettingsScreen from "#/routes/app-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";
import ProfilesService from "#/api/profiles-service/profiles-service.api";

const activeBackendState = vi.hoisted(() => ({
  kind: "local" as "local" | "cloud",
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { kind: activeBackendState.kind },
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
  return render(<AppSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });
}

describe("AppSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    activeBackendState.kind = "local";
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
});
