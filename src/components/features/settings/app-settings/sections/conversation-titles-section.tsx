import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { formatModelNameForDisplay } from "#/utils/format-model-name";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

const AUTOMATIC_TITLE_LLM_PROFILE_KEY = "__automatic__";

/**
 * Chooses which LLM profile generates conversation titles. Section-owned save —
 * persists only `title_llm_profile`.
 */
export function ConversationTitlesSection() {
  const { t } = useTranslation("openhands");
  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings } = useSettings();
  const { data: llmProfiles } = useLlmProfiles();

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
              model: formatModelNameForDisplay(profile.model) ?? profile.model,
            })
          : profile.name,
      })) ?? []),
    ],
    [llmProfiles?.profiles, t],
  );

  if (!settings) return null;

  const formAction = () => {
    saveSettings(
      { title_llm_profile: selectedTitleLlmProfile },
      {
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
        },
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
        onSettled: () => {
          setTitleLlmProfileInput(undefined);
        },
      },
    );
  };

  const formIsClean = selectedTitleLlmProfile === storedTitleLlmProfile;

  return (
    <form
      data-testid="app-settings-conversation-titles-section"
      action={formAction}
      className="border-t border-[var(--oh-border)] pt-6 mt-2"
    >
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
        selectedKey={selectedTitleLlmProfile ?? AUTOMATIC_TITLE_LLM_PROFILE_KEY}
        onSelectionChange={(key) => {
          const value = key?.toString();
          setTitleLlmProfileInput(
            !value || value === AUTOMATIC_TITLE_LLM_PROFILE_KEY ? null : value,
          );
        }}
      />
      <NavigationLink
        to="/settings/llm"
        className="mt-3 inline-block text-sm text-primary hover:underline"
      >
        {t(I18nKey.SETTINGS$MANAGE_LLM_PROFILES)}
      </NavigationLink>

      <div className="flex justify-start pt-4">
        <BrandButton
          testId="app-settings-conversation-titles-submit"
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
