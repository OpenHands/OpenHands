import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import GeneralSettingsScreen from "#/routes/general-settings";
import { Settings } from "#/types/settings";

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    conversation_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
      ...overrides.conversation_settings,
    },
    conversation_settings_schema:
      overrides.conversation_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.conversation_settings_schema,
  };
}

function renderGeneralSettingsScreen() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(<GeneralSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("GeneralSettingsScreen", () => {
  it("renders the max_iterations field from the schema with the default value", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderGeneralSettingsScreen();

    await screen.findByTestId("general-settings-screen");

    // max_iterations is the only field in the general section. The page has
    // no critical-prominence fields, so it floors its view at "advanced" and
    // renders the field directly (same behaviour as the memory page).
    const input = await screen.findByTestId("sdk-settings-max_iterations");
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("type", "number");
    // Default from the persisted conversation settings (mock default 500).
    expect(input).toHaveValue(500);
  });

  it("reflects a custom persisted max_iterations value", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        conversation_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
          max_iterations: 2000,
        },
      }),
    );

    renderGeneralSettingsScreen();

    await screen.findByTestId("general-settings-screen");

    const input = await screen.findByTestId("sdk-settings-max_iterations");
    expect(input).toHaveValue(2000);
  });

  it("saves an edited value through the conversation settings diff", async () => {
    let persistedSettings = buildSettings();

    vi.spyOn(SettingsService, "getSettings").mockImplementation(async () =>
      structuredClone(persistedSettings),
    );
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementation(async (payload) => {
        const diff = payload.conversation_settings_diff as Record<
          string,
          unknown
        >;
        if (typeof diff.max_iterations === "number") {
          persistedSettings = buildSettings({
            conversation_settings: {
              ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
              max_iterations: diff.max_iterations,
            },
          });
        }
        return true;
      });

    renderGeneralSettingsScreen();

    await screen.findByTestId("general-settings-screen");

    const input = await screen.findByTestId("sdk-settings-max_iterations");
    await userEvent.clear(input);
    await userEvent.type(input, "2000");
    await userEvent.click(screen.getByTestId("save-button"));

    // The value propagates through the conversation-settings diff untouched —
    // values above the old hard cap (500) must be accepted.
    await waitFor(() => {
      expect(saveSettingsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          conversation_settings_diff: { max_iterations: 2000 },
        }),
      );
    });

    // Settings refetch after save reflects the persisted value.
    await waitFor(() => {
      expect(screen.getByTestId("sdk-settings-max_iterations")).toHaveValue(
        2000,
      );
    });
  });

  it("keeps a large value (above the old 500 default) without artificial capping", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        conversation_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
          max_iterations: 5000,
        },
      }),
    );

    renderGeneralSettingsScreen();

    await screen.findByTestId("general-settings-screen");

    const input = await screen.findByTestId("sdk-settings-max_iterations");
    expect(input).toHaveValue(5000);
  });
});
