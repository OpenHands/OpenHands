import React from "react";
import { useTranslation } from "react-i18next";
import { usePostHog } from "posthog-js/react";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { AvailableLanguages } from "#/i18n";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { LanguageInput } from "#/components/features/settings/app-settings/language-input";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { AppSettingsInputsSkeleton } from "#/components/features/settings/app-settings/app-settings-inputs-skeleton";
import { useConfig } from "#/hooks/query/use-config";
import { parseMaxBudgetPerTask } from "#/utils/settings-utils";

function AppSettingsScreen() {
  const posthog = usePostHog();
  const { t } = useTranslation();

  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings, isLoading } = useSettings();
  const { data: config } = useConfig();

  const [languageInputHasChanged, setLanguageInputHasChanged] =
    React.useState(false);
  const [analyticsSwitchHasChanged, setAnalyticsSwitchHasChanged] =
    React.useState(false);
  const [
    soundNotificationsSwitchHasChanged,
    setSoundNotificationsSwitchHasChanged,
  ] = React.useState(false);
  const [
    proactiveConversationsSwitchHasChanged,
    setProactiveConversationsSwitchHasChanged,
  ] = React.useState(false);
  const [
    solvabilityAnalysisSwitchHasChanged,
    setSolvabilityAnalysisSwitchHasChanged,
  ] = React.useState(false);
  const [maxBudgetPerTaskHasChanged, setMaxBudgetPerTaskHasChanged] =
    React.useState(false);
  const [gitUserNameHasChanged, setGitUserNameHasChanged] =
    React.useState(false);
  const [gitUserEmailHasChanged, setGitUserEmailHasChanged] =
    React.useState(false);

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

    const enableProactiveConversations =
      formData.get("enable-proactive-conversations-switch")?.toString() ===
      "on";

    const enableSolvabilityAnalysis =
      formData.get("enable-solvability-analysis-switch")?.toString() === "on";

    const maxBudgetPerTaskValue = formData
      .get("max-budget-per-task-input")
      ?.toString();
    const maxBudgetPerTask = parseMaxBudgetPerTask(maxBudgetPerTaskValue || "");

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
        enable_proactive_conversation_starters: enableProactiveConversations,
        enable_solvability_analysis: enableSolvabilityAnalysis,
        max_budget_per_task: maxBudgetPerTask,
        git_user_name: gitUserName,
        git_user_email: gitUserEmail,
      },
      {
        onSuccess: () => {
          handleCaptureConsent(posthog, enableAnalytics);
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
          setProactiveConversationsSwitchHasChanged(false);
          setMaxBudgetPerTaskHasChanged(false);
          setGitUserNameHasChanged(false);
          setGitUserEmailHasChanged(false);
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

  const checkIfProactiveConversationsSwitchHasChanged = (checked: boolean) => {
    const currentProactiveConversations =
      !!settings?.enable_proactive_conversation_starters;
    setProactiveConversationsSwitchHasChanged(
      checked !== currentProactiveConversations,
    );
  };

  const checkIfSolvabilityAnalysisSwitchHasChanged = (checked: boolean) => {
    const currentSolvabilityAnalysis = !!settings?.enable_solvability_analysis;
    setSolvabilityAnalysisSwitchHasChanged(
      checked !== currentSolvabilityAnalysis,
    );
  };

  const checkIfMaxBudgetPerTaskHasChanged = (value: string) => {
    const newValue = parseMaxBudgetPerTask(value);
    const currentValue = settings?.max_budget_per_task;
    setMaxBudgetPerTaskHasChanged(newValue !== currentValue);
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
    !proactiveConversationsSwitchHasChanged &&
    !solvabilityAnalysisSwitchHasChanged &&
    !maxBudgetPerTaskHasChanged &&
    !gitUserNameHasChanged &&
    !gitUserEmailHasChanged;

  const shouldBeLoading = !settings || isLoading || isPending;

  return (
    <form
      data-testid="app-settings-screen"
      action={formAction}
      className="flex flex-col h-full justify-between"
    >
      {shouldBeLoading && <AppSettingsInputsSkeleton />}
      {!shouldBeLoading && (
        <div className="flex flex-col gap-6">
          <LanguageInput
            name="language-input"
            defaultKey={settings.language}
            onChange={checkIfLanguageInputHasChanged}
          />

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

          {config?.APP_MODE === "saas" && (
            <SettingsSwitch
              testId="enable-proactive-conversations-switch"
              name="enable-proactive-conversations-switch"
              defaultIsToggled={
                !!settings.enable_proactive_conversation_starters
              }
              onToggle={checkIfProactiveConversationsSwitchHasChanged}
            >
              {t(I18nKey.SETTINGS$PROACTIVE_CONVERSATION_STARTERS)}
            </SettingsSwitch>
          )}

          {config?.APP_MODE === "saas" && (
            <SettingsSwitch
              testId="enable-solvability-analysis-switch"
              name="enable-solvability-analysis-switch"
              defaultIsToggled={!!settings.enable_solvability_analysis}
              onToggle={checkIfSolvabilityAnalysisSwitchHasChanged}
            >
              {t(I18nKey.SETTINGS$SOLVABILITY_ANALYSIS)}
            </SettingsSwitch>
          )}

          {!settings?.v1_enabled && (
            <SettingsInput
              testId="max-budget-per-task-input"
              name="max-budget-per-task-input"
              type="number"
              label={t(I18nKey.SETTINGS$MAX_BUDGET_PER_CONVERSATION)}
              defaultValue={settings.max_budget_per_task?.toString() || ""}
              onChange={checkIfMaxBudgetPerTaskHasChanged}
              placeholder={t(I18nKey.SETTINGS$MAXIMUM_BUDGET_USD)}
              min={1}
              step={1}
              className="w-full max-w-[680px]" // Match the width of the language field
            />
          )}

          <div className="border-t border-t-tertiary pt-6 mt-2">
            <h3 className="text-lg font-medium mb-2">
              {t(I18nKey.SETTINGS$GIT_SETTINGS)}
            </h3>
            <p className="text-xs mb-4">
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
                placeholder="Username for git commits"
                className="w-full max-w-[680px]"
              />
              <SettingsInput
                testId="git-user-email-input"
                name="git-user-email-input"
                type="email"
                label={t(I18nKey.SETTINGS$GIT_EMAIL)}
                defaultValue={settings.git_user_email || ""}
                onChange={checkIfGitUserEmailHasChanged}
                placeholder="Email for git commits"
                className="w-full max-w-[680px]"
              />
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-6 p-6 justify-end">
        <BrandButton
          testId="submit-button"
          variant="primary"
          type="submit"
          isDisabled={isPending || formIsClean}
        >
          {!isPending && t("SETTINGS$SAVE_CHANGES")}
          {isPending && t("SETTINGS$SAVING")}
        </BrandButton>
      </div>
    </form>
  );
}

export default AppSettingsScreen;
