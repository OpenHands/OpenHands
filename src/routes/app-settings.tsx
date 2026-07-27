import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { AvailableLanguages } from "#/i18n";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { setTelemetryConsent } from "#/services/telemetry";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { LanguageInput } from "#/components/features/settings/app-settings/language-input";
import { ThemeInput } from "#/components/features/settings/app-settings/theme-input";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { AppSettingsInputsSkeleton } from "#/components/features/settings/app-settings/app-settings-inputs-skeleton";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";

const AUTOMATIC_TITLE_LLM_PROFILE_KEY = "__automatic__";

export function AppSettingsScreen() {
  const { t } = useTranslation("openhands");

  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings, isLoading } = useSettings();
  const { data: llmProfiles, isLoading: areLlmProfilesLoading } =
    useLlmProfiles();

  const [languageInputHasChanged, setLanguageInputHasChanged] =
    React.useState(false);
  const [analyticsSwitchHasChanged, setAnalyticsSwitchHasChanged] =
    React.useState(false);
  const [
    soundNotificationsSwitchHasChanged,
    setSoundNotificationsSwitchHasChanged,
  ] = React.useState(false);
  const [gitUserNameHasChanged, setGitUserNameHasChanged] =
    React.useState(false);
  const [gitUserEmailHasChanged, setGitUserEmailHasChanged] =
    React.useState(false);
  const [titleLlmProfileInput, setTitleLlmProfileInput] = React.useState<
    string | null | undefined
  >(undefined);

  const storedTitleLlmProfile = React.useMemo(() => {
    const preference = settings?.title_llm_profile ?? null;
    if (!preference || !llmProfiles) return preference;
    return llmProfiles.profiles.some((profile) => profile.name === preference)
      ? preference
      : null;
  }, [llmProfiles, settings?.title_llm_profile]);
  const selectedTitleLlmProfile =
    titleLlmProfileInput === undefined
      ? storedTitleLlmProfile
      : titleLlmProfileInput;
  const titleLlmProfileItems = React.useMemo(
    () => [
      {
        key: AUTOMATIC_TITLE_LLM_PROFILE_KEY,
        label: t(I18nKey.SETTINGS$TITLE_GENERATION_AUTOMATIC),
      },
      ...(llmProfiles?.profiles.map((profile) => ({
        key: profile.name,
        label: profile.model
          ? t(I18nKey.SETTINGS$TITLE_GENERATION_PROFILE_OPTION, {
              name: profile.name,
              model: profile.model,
            })
          : profile.name,
      })) ?? []),
    ],
    [llmProfiles?.profiles, t],
  );

  const formAction = (formData: FormData) => {
    const languageLabel = formData.get("language-input")?.toString();
    const languageValue = AvailableLanguages.find(
      ({ label }) => label === languageLabel,
    )?.value;
    const language = languageValue || DEFAULT_SETTINGS.language;

    const enableAnalytics =
      formData.get("enable-analytics-switch")?.toString() === "on";
    const enableSoundNotifications =
      formData.get("enable-sound-notifications-switch")?.toString() === "on";

    const gitUserName =
      formData.get("git-user-name-input")?.toString() ||
      DEFAULT_SETTINGS.git_user_name;
    const gitUserEmail =
      formData.get("git-user-email-input")?.toString() ||
      DEFAULT_SETTINGS.git_user_email;

    saveSettings(
      {
        language,
        user_consents_to_analytics: enableAnalytics,
        enable_sound_notifications: enableSoundNotifications,
        git_user_name: gitUserName,
        git_user_email: gitUserEmail,
        title_llm_profile: selectedTitleLlmProfile,
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
          setGitUserNameHasChanged(false);
          setGitUserEmailHasChanged(false);
          setTitleLlmProfileInput(undefined);
        },
      },
    );
  };

  const checkIfLanguageInputHasChanged = (value: string) => {
    const selectedLanguage = AvailableLanguages.find(
      ({ label: langValue }) => langValue === value,
    )?.label;
    const currentLanguage = AvailableLanguages.find(
      ({ value: langValue }) => langValue === settings?.language,
    )?.label;

    setLanguageInputHasChanged(selectedLanguage !== currentLanguage);
  };

  const checkIfAnalyticsSwitchHasChanged = (checked: boolean) => {
    // Treat null as true since analytics is opt-in by default
    const currentAnalytics = settings?.user_consents_to_analytics ?? true;
    setAnalyticsSwitchHasChanged(checked !== currentAnalytics);
  };

  const checkIfSoundNotificationsSwitchHasChanged = (checked: boolean) => {
    const currentSoundNotifications = !!settings?.enable_sound_notifications;
    setSoundNotificationsSwitchHasChanged(
      checked !== currentSoundNotifications,
    );
  };

  const checkIfGitUserNameHasChanged = (value: string) => {
    const currentValue = settings?.git_user_name;
    setGitUserNameHasChanged(value !== currentValue);
  };

  const checkIfGitUserEmailHasChanged = (value: string) => {
    const currentValue = settings?.git_user_email;
    setGitUserEmailHasChanged(value !== currentValue);
  };

  const formIsClean =
    !languageInputHasChanged &&
    !analyticsSwitchHasChanged &&
    !soundNotificationsSwitchHasChanged &&
    selectedTitleLlmProfile === storedTitleLlmProfile &&
    !gitUserNameHasChanged &&
    !gitUserEmailHasChanged;

  const shouldBeLoading =
    !settings || isLoading || areLlmProfilesLoading || isPending;

  return (
    <form
      data-testid="app-settings-screen"
      action={formAction}
      className="flex flex-col gap-6"
    >
      {shouldBeLoading && <AppSettingsInputsSkeleton />}
      {!shouldBeLoading && (
        <div className="flex flex-col gap-6">
          <LanguageInput
            name="language-input"
            defaultKey={settings.language}
            onChange={checkIfLanguageInputHasChanged}
          />

          <ThemeInput />

          <SettingsSwitch
            testId="enable-analytics-switch"
            name="enable-analytics-switch"
            defaultIsToggled={settings.user_consents_to_analytics ?? true}
            onToggle={checkIfAnalyticsSwitchHasChanged}
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

          <div className="border-t border-[var(--oh-border)] pt-6 mt-2">
            <h3 className="text-lg font-medium mb-2">
              {t(I18nKey.SETTINGS$CONVERSATION_TITLES)}
            </h3>
            <p className="mb-4 text-sm leading-5 text-tertiary-light">
              {t(I18nKey.SETTINGS$TITLE_GENERATION_DESCRIPTION)}
            </p>
            <SettingsDropdownInput
              testId="title-llm-profile-input"
              name="title-llm-profile-input"
              label={t(I18nKey.SETTINGS$TITLE_GENERATION_MODEL)}
              items={titleLlmProfileItems}
              selectedKey={
                selectedTitleLlmProfile ?? AUTOMATIC_TITLE_LLM_PROFILE_KEY
              }
              onSelectionChange={(key) => {
                const value = key?.toString();
                setTitleLlmProfileInput(
                  !value || value === AUTOMATIC_TITLE_LLM_PROFILE_KEY
                    ? null
                    : value,
                );
              }}
            />
            <NavigationLink
              to="/settings/llm"
              className="mt-3 inline-block text-sm text-primary hover:underline"
            >
              {t(I18nKey.SETTINGS$MANAGE_LLM_PROFILES)}
            </NavigationLink>
          </div>

          <div className="border-t border-[var(--oh-border)] pt-6 mt-2">
            <h3 className="text-lg font-medium mb-2">
              {t(I18nKey.SETTINGS$GIT_SETTINGS)}
            </h3>
            <p className="mb-4 text-sm leading-5 text-tertiary-light">
              {t(I18nKey.SETTINGS$GIT_SETTINGS_DESCRIPTION)}
            </p>
            <div className="flex flex-col gap-6">
              <SettingsInput
                testId="git-user-name-input"
                name="git-user-name-input"
                type="text"
                label={t(I18nKey.SETTINGS$GIT_USERNAME)}
                defaultValue={settings.git_user_name || ""}
                onChange={checkIfGitUserNameHasChanged}
                placeholder={t(I18nKey.SETTINGS$GIT_USERNAME_PLACEHOLDER)}
                className="w-full min-w-0"
              />
              <SettingsInput
                testId="git-user-email-input"
                name="git-user-email-input"
                type="email"
                label={t(I18nKey.SETTINGS$GIT_EMAIL)}
                defaultValue={settings.git_user_email || ""}
                onChange={checkIfGitUserEmailHasChanged}
                placeholder={t(I18nKey.SETTINGS$GIT_EMAIL_PLACEHOLDER)}
                className="w-full min-w-0"
              />
            </div>
            <div className="flex justify-start pt-4">
              <BrandButton
                testId="submit-button"
                variant="primary"
                type="submit"
                isDisabled={isPending || formIsClean}
              >
                {!isPending && t(I18nKey.SETTINGS$SAVE_CHANGES)}
                {isPending && t(I18nKey.SETTINGS$SAVING)}
              </BrandButton>
            </div>
          </div>
        </div>
      )}
    </form>
  );
}

export default AppSettingsScreen;
