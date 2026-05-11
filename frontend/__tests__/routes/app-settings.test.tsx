import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import AppSettingsScreen, { clientLoader } from "#/routes/app-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { AvailableLanguages } from "#/i18n";
import * as CaptureConsent from "#/utils/handle-capture-consent";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { createMockWebClientConfig } from "#/mocks/settings-handlers";

beforeEach(() => {
  useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
});

const renderAppSettingsScreen = () =>
  render(<AppSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });

describe("clientLoader permission checks", () => {
  it("should export a clientLoader for route protection", () => {
    // This test verifies the clientLoader is exported (for consistency with other routes)
    expect(clientLoader).toBeDefined();
    expect(typeof clientLoader).toBe("function");
  });
});

describe("Content", () => {
  it("should render the screen", () => {
    renderAppSettingsScreen();
    screen.getByTestId("app-settings-screen");
  });

  it("should render the correct default values", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      language: "no",
      user_consents_to_analytics: true,
      enable_sound_notifications: true,
    });

    renderAppSettingsScreen();

    await waitFor(() => {
      const language = screen.getByTestId("language-input");
      const analytics = screen.getByTestId("enable-analytics-switch");
      const sound = screen.getByTestId("enable-sound-notifications-switch");

      expect(language).toHaveValue("Norsk");
      expect(analytics).toBeChecked();
      expect(sound).toBeChecked();
    });
  });

  it("should render analytics toggle as enabled when server returns null (opt-in by default)", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      user_consents_to_analytics: null,
    });

    renderAppSettingsScreen();

    await waitFor(() => {
      const analytics = screen.getByTestId("enable-analytics-switch");
      expect(analytics).toBeChecked();
    });
  });

  it("should render the language options", async () => {
    renderAppSettingsScreen();

    const language = await screen.findByTestId("language-input");
    await userEvent.click(language);

    AvailableLanguages.forEach((lang) => {
      const option = screen.getByText(lang.label);
      expect(option).toBeInTheDocument();
    });
  });
});

describe("Form submission", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should submit the form with the correct values", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    renderAppSettingsScreen();

    const language = await screen.findByTestId("language-input");
    const analytics = await screen.findByTestId("enable-analytics-switch");
    const sound = await screen.findByTestId(
      "enable-sound-notifications-switch",
    );

    expect(language).toHaveValue("English");
    expect(analytics).not.toBeChecked();
    expect(sound).not.toBeChecked();

    // change language
    await userEvent.click(language);
    const norsk = screen.getByText("Norsk");
    await userEvent.click(norsk);
    expect(language).toHaveValue("Norsk");

    // toggle options
    await userEvent.click(analytics);
    expect(analytics).toBeChecked();
    await userEvent.click(sound);
    expect(sound).toBeChecked();

    // submit the form
    const submit = await screen.findByTestId("submit-button");
    await userEvent.click(submit);
    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        language: "no",
        user_consents_to_analytics: true,
        enable_sound_notifications: true,
      }),
    );
  });

  it("should only enable the submit button when there are changes", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    renderAppSettingsScreen();

    const submit = await screen.findByTestId("submit-button");
    expect(submit).toBeDisabled();

    // Language check
    const language = await screen.findByTestId("language-input");
    await userEvent.click(language);
    const norsk = screen.getByText("Norsk");
    await userEvent.click(norsk);
    expect(submit).not.toBeDisabled();

    await userEvent.click(language);
    const english = screen.getByText("English");
    await userEvent.click(english);
    expect(submit).toBeDisabled();

    // Analytics check
    const analytics = await screen.findByTestId("enable-analytics-switch");
    await userEvent.click(analytics);
    expect(submit).not.toBeDisabled();

    await userEvent.click(analytics);
    expect(submit).toBeDisabled();

    // Sound check
    const sound = await screen.findByTestId(
      "enable-sound-notifications-switch",
    );
    await userEvent.click(sound);
    expect(submit).not.toBeDisabled();

    await userEvent.click(sound);
    expect(submit).toBeDisabled();
  });

  it("should call handleCaptureConsents with true when the analytics switch is toggled", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    const handleCaptureConsentsSpy = vi.spyOn(
      CaptureConsent,
      "handleCaptureConsent",
    );

    renderAppSettingsScreen();

    const analytics = await screen.findByTestId("enable-analytics-switch");
    const submit = await screen.findByTestId("submit-button");

    await userEvent.click(analytics);
    await userEvent.click(submit);

    await waitFor(() =>
      expect(handleCaptureConsentsSpy).toHaveBeenCalledWith(
        expect.anything(),
        true,
      ),
    );
  });

  it("should call handleCaptureConsents with false when the analytics switch is toggled", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      user_consents_to_analytics: true,
    });

    const handleCaptureConsentsSpy = vi.spyOn(
      CaptureConsent,
      "handleCaptureConsent",
    );

    renderAppSettingsScreen();

    const analytics = await screen.findByTestId("enable-analytics-switch");
    const submit = await screen.findByTestId("submit-button");

    await userEvent.click(analytics);
    await userEvent.click(submit);

    await waitFor(() =>
      expect(handleCaptureConsentsSpy).toHaveBeenCalledWith(
        expect.anything(),
        false,
      ),
    );
  });

  // flaky test
  it.skip("should disable the button when submitting changes", async () => {
    renderAppSettingsScreen();

    const submit = await screen.findByTestId("submit-button");
    expect(submit).toBeDisabled();

    const sound = await screen.findByTestId(
      "enable-sound-notifications-switch",
    );
    await userEvent.click(sound);
    expect(submit).not.toBeDisabled();

    // submit the form
    await userEvent.click(submit);

    expect(submit).toHaveTextContent("Saving...");
    expect(submit).toBeDisabled();

    await waitFor(() => expect(submit).toHaveTextContent("Save"));
  });

  it("should disable the button after submitting changes", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    renderAppSettingsScreen();

    const submit = await screen.findByTestId("submit-button");
    expect(submit).toBeDisabled();

    const sound = await screen.findByTestId(
      "enable-sound-notifications-switch",
    );
    await userEvent.click(sound);
    expect(submit).not.toBeDisabled();

    // submit the form
    await userEvent.click(submit);
    expect(saveSettingsSpy).toHaveBeenCalled();

    await waitFor(() => expect(submit).toBeDisabled());
  });
});

describe("Status toasts", () => {
  it("should call displaySuccessToast when the settings are saved", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    const displaySuccessToastSpy = vi.spyOn(
      ToastHandlers,
      "displaySuccessToast",
    );

    renderAppSettingsScreen();

    // Toggle setting to change
    const sound = await screen.findByTestId(
      "enable-sound-notifications-switch",
    );
    await userEvent.click(sound);

    const submit = await screen.findByTestId("submit-button");
    await userEvent.click(submit);

    expect(saveSettingsSpy).toHaveBeenCalled();
    await waitFor(() => expect(displaySuccessToastSpy).toHaveBeenCalled());
  });

  it("should call displayErrorToast when the settings fail to save", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

    saveSettingsSpy.mockRejectedValue(new Error("Failed to save settings"));

    renderAppSettingsScreen();

    // Toggle setting to change
    const sound = await screen.findByTestId(
      "enable-sound-notifications-switch",
    );
    await userEvent.click(sound);

    const submit = await screen.findByTestId("submit-button");
    await userEvent.click(submit);

    expect(saveSettingsSpy).toHaveBeenCalled();
    expect(displayErrorToastSpy).toHaveBeenCalled();
  });
});

describe("Stay logged in switch", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const mockSaasConfig = () => {
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(
      createMockWebClientConfig({ app_mode: "saas" }),
    );
  };

  it("should not render the stay-logged-in switch in oss mode", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
    // oss mode by default (no spy on OptionService.getConfig)
    renderAppSettingsScreen();

    // Wait for settings to load
    await screen.findByTestId("enable-analytics-switch");

    expect(
      screen.queryByTestId("stay-logged-in-switch"),
    ).not.toBeInTheDocument();
  });

  it("should render the stay-logged-in switch in saas mode", async () => {
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );

    renderAppSettingsScreen();

    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");
    expect(stayLoggedIn).toBeInTheDocument();
  });

  it("should default stay-logged-in to checked when settings.stay_logged_in is true", async () => {
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      stay_logged_in: true,
    });

    renderAppSettingsScreen();

    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");
    expect(stayLoggedIn).toBeChecked();
  });

  it("should render stay-logged-in as unchecked when settings.stay_logged_in is false", async () => {
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      stay_logged_in: false,
    });

    renderAppSettingsScreen();

    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");
    expect(stayLoggedIn).not.toBeChecked();
  });

  it("should enable the submit button when stay-logged-in is toggled", async () => {
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      stay_logged_in: true,
    });

    renderAppSettingsScreen();

    const submit = await screen.findByTestId("submit-button");
    expect(submit).toBeDisabled();

    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");
    await userEvent.click(stayLoggedIn);
    expect(submit).not.toBeDisabled();
  });

  it("should disable the submit button when stay-logged-in is toggled back to original value", async () => {
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      stay_logged_in: true,
    });

    renderAppSettingsScreen();

    const submit = await screen.findByTestId("submit-button");
    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");

    await userEvent.click(stayLoggedIn);
    expect(submit).not.toBeDisabled();

    await userEvent.click(stayLoggedIn);
    expect(submit).toBeDisabled();
  });

  it("should submit stay_logged_in: false when switch is turned off", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      stay_logged_in: true,
    });

    renderAppSettingsScreen();

    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");
    await userEvent.click(stayLoggedIn);

    const submit = await screen.findByTestId("submit-button");
    await userEvent.click(submit);

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({ stay_logged_in: false }),
      expect.anything(),
    );
  });

  it("should submit stay_logged_in: true when switch is turned on", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      stay_logged_in: false,
    });

    renderAppSettingsScreen();

    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");
    await userEvent.click(stayLoggedIn);

    const submit = await screen.findByTestId("submit-button");
    await userEvent.click(submit);

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({ stay_logged_in: true }),
      expect.anything(),
    );
  });

  it("should reset stayLoggedInSwitchHasChanged to false after successful save", async () => {
    mockSaasConfig();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      stay_logged_in: true,
    });

    renderAppSettingsScreen();

    const stayLoggedIn = await screen.findByTestId("stay-logged-in-switch");
    await userEvent.click(stayLoggedIn);

    const submit = await screen.findByTestId("submit-button");
    expect(submit).not.toBeDisabled();

    await userEvent.click(submit);

    await waitFor(() => expect(submit).toBeDisabled());
  });
});
