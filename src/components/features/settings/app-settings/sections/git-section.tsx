import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/**
 * Git identity used for commits (`git_user_name` / `git_user_email`).
 * Section-owned save — persists only these two fields.
 */
export function GitSection() {
  const { t } = useTranslation("openhands");
  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings } = useSettings();

  const [gitUserNameHasChanged, setGitUserNameHasChanged] =
    React.useState(false);
  const [gitUserEmailHasChanged, setGitUserEmailHasChanged] =
    React.useState(false);

  if (!settings) return null;

  const formAction = (formData: FormData) => {
    const gitUserName =
      formData.get("git-user-name-input")?.toString() ||
      DEFAULT_SETTINGS.git_user_name;
    const gitUserEmail =
      formData.get("git-user-email-input")?.toString() ||
      DEFAULT_SETTINGS.git_user_email;

    saveSettings(
      { git_user_name: gitUserName, git_user_email: gitUserEmail },
      {
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
        },
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
        onSettled: () => {
          setGitUserNameHasChanged(false);
          setGitUserEmailHasChanged(false);
        },
      },
    );
  };

  const checkIfGitUserNameHasChanged = (value: string) => {
    setGitUserNameHasChanged(value !== settings.git_user_name);
  };

  const checkIfGitUserEmailHasChanged = (value: string) => {
    setGitUserEmailHasChanged(value !== settings.git_user_email);
  };

  const formIsClean = !gitUserNameHasChanged && !gitUserEmailHasChanged;

  return (
    <form
      data-testid="app-settings-git-section"
      action={formAction}
      className="border-t border-[var(--oh-border)] pt-6 mt-2"
    >
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
          testId="app-settings-git-submit"
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
