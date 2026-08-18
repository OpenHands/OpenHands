import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { AvailableLanguages } from "#/i18n";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { setTelemetryConsent } from "#/services/telemetry";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { I18nKey } from "#/i18n/declaration";
import { LanguageInput } from "#/components/features/settings/app-settings/language-input";
import { ThemeInput } from "#/components/features/settings/app-settings/theme-input";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { useSettingsContext } from "#/settings/use-settings-context";

/**
 * General application preferences: language, theme, analytics consent, and
 * sound notifications. Section-owned save — it persists only these fields.
 *
 * Analytics consent is controlled by TOS on cloud backends, so the toggle is
 * forced on/disabled and the field is omitted from the save payload there
 * (mirrors the behaviour the monolithic Application page had).
 */
export function GeneralSection() {
  const { t } = useTranslation("openhands");
  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings } = useSettings();
  const { backendKind } = useSettingsContext();
  const isCloudBackend = backendKind === "cloud";

  const [languageInputHasChanged, setLanguageInputHasChanged] =
    React.useState(false);
  const [analyticsSwitchHasChanged, setAnalyticsSwitchHasChanged] =
    React.useState(false);
  const [
    soundNotificationsSwitchHasChanged,
    setSoundNotificationsSwitchHasChanged,
  ] = React.useState(false);

  if (!settings) return null;

  const formAction = (formData: FormData) => {
    const languageLabel = formData.get("language-input")?.toString();
    const languageValue = AvailableLanguages.find(
      ({ label }) => label === languageLabel,
    )?.value;
    const language = languageValue || DEFAULT_SETTINGS.language;

    const enableAnalytics = isCloudBackend
      ? true
      : formData.get("enable-analytics-switch")?.toString() === "on";
    const enableSoundNotifications =
      formData.get("enable-sound-notifications-switch")?.toString() === "on";

    saveSettings(
      {
        language,
        ...(!isCloudBackend && { user_consents_to_analytics: enableAnalytics }),
        enable_sound_notifications: enableSoundNotifications,
      },
      {
        onSuccess: () => {
          void setTelemetryConsent(enableAnalytics ? "granted" : "denied");
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
        },
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
        onSettled: () => {
          setLanguageInputHasChanged(false);
          setAnalyticsSwitchHasChanged(false);
          setSoundNotificationsSwitchHasChanged(false);
        },
      },
    );
  };

  const checkIfLanguageInputHasChanged = (value: string) => {
    const selectedLanguage = AvailableLanguages.find(
      ({ label: langValue }) => langValue === value,
    )?.label;
    const currentLanguage = AvailableLanguages.find(
      ({ value: langValue }) => langValue === settings.language,
    )?.label;
    setLanguageInputHasChanged(selectedLanguage !== currentLanguage);
  };

  const checkIfAnalyticsSwitchHasChanged = (checked: boolean) => {
    // Treat null as true since analytics is opt-in by default
    const currentAnalytics = settings.user_consents_to_analytics ?? true;
    setAnalyticsSwitchHasChanged(checked !== currentAnalytics);
  };

  const checkIfSoundNotificationsSwitchHasChanged = (checked: boolean) => {
    const currentSoundNotifications = !!settings.enable_sound_notifications;
    setSoundNotificationsSwitchHasChanged(
      checked !== currentSoundNotifications,
    );
  };

  const formIsClean =
    !languageInputHasChanged &&
    !analyticsSwitchHasChanged &&
    !soundNotificationsSwitchHasChanged;

  return (
    <form
      data-testid="app-settings-general-section"
      action={formAction}
      className="flex flex-col gap-6"
    >
      <LanguageInput
        name="language-input"
        defaultKey={settings.language}
        onChange={checkIfLanguageInputHasChanged}
      />

      <ThemeInput />

      <SettingsSwitch
        testId="enable-analytics-switch"
        name={isCloudBackend ? undefined : "enable-analytics-switch"}
        defaultIsToggled={
          isCloudBackend ? true : (settings.user_consents_to_analytics ?? true)
        }
        isToggled={isCloudBackend ? true : undefined}
        isDisabled={isCloudBackend}
        onToggle={isCloudBackend ? undefined : checkIfAnalyticsSwitchHasChanged}
      >
        {t(I18nKey.ANALYTICS$SEND_ANONYMOUS_DATA)}
      </SettingsSwitch>

      <SettingsSwitch
        testId="enable-sound-notifications-switch"
        name="enable-sound-notifications-switch"
        defaultIsToggled={!!settings.enable_sound_notifications}
        onToggle={checkIfSoundNotificationsSwitchHasChanged}
      >
        {t(I18nKey.SETTINGS$SOUND_NOTIFICATIONS)}
      </SettingsSwitch>

      <div className="flex justify-start">
        <BrandButton
          testId="app-settings-general-submit"
          variant="primary"
          type="submit"
          isDisabled={isPending || formIsClean}
        >
          {!isPending && t(I18nKey.SETTINGS$SAVE_CHANGES)}
          {isPending && t(I18nKey.SETTINGS$SAVING)}
        </BrandButton>
      </div>
    </form>
  );
}
