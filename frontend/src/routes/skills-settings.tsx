import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useSkills } from "#/hooks/query/use-skills";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

function SkillsSettingsScreen() {
  const { t } = useTranslation();

  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: skills, isLoading: skillsLoading } = useSkills();

  // Local state: set of skill names the user has toggled off
  const [disabledSet, setDisabledSet] = React.useState<Set<string>>(new Set());
  const [hasChanges, setHasChanges] = React.useState(false);

  // Sync local state with server settings when data first arrives
  React.useEffect(() => {
    if (settings?.disabled_skills) {
      setDisabledSet(new Set(settings.disabled_skills));
    }
  }, [settings?.disabled_skills]);

  const handleToggle = (skillName: string, enabled: boolean) => {
    setDisabledSet((prev) => {
      const next = new Set(prev);
      if (enabled) {
        next.delete(skillName);
      } else {
        next.add(skillName);
      }
      return next;
    });
    setHasChanges(true);
  };

  const handleSave = () => {
    saveSettings(
      { disabled_skills: Array.from(disabledSet) },
      {
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
          setHasChanges(false);
        },
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
      },
    );
  };

  const isLoading = settingsLoading || skillsLoading || !settings;

  return (
    <div data-testid="skills-settings-screen" className="flex flex-col h-full">
      <p className="text-xs mb-4">{t(I18nKey.SETTINGS$SKILLS_DESCRIPTION)}</p>

      <div className="flex-1 overflow-auto custom-scrollbar-always">
        {isLoading && (
          <div className="flex flex-col gap-4">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-8 w-64 rounded bg-tertiary animate-pulse"
              />
            ))}
          </div>
        )}

        {!isLoading && (!skills || skills.length === 0) && (
          <p className="text-sm text-tertiary">
            {t(I18nKey.SETTINGS$SKILLS_NO_SKILLS)}
          </p>
        )}

        {!isLoading && skills && skills.length > 0 && (
          <div className="flex flex-col gap-4">
            {skills.map((skill) => (
              <div key={skill.name} className="flex flex-col gap-0.5">
                <SettingsSwitch
                  testId={`skill-toggle-${skill.name}`}
                  isToggled={!disabledSet.has(skill.name)}
                  onToggle={(enabled) => handleToggle(skill.name, enabled)}
                >
                  {skill.name}
                </SettingsSwitch>
                {skill.triggers && skill.triggers.length > 0 && (
                  <span className="text-xs text-neutral-500 ml-14">
                    {t(I18nKey.SETTINGS$SKILLS_TRIGGERS, {
                      triggers: skill.triggers.join(", "),
                      interpolation: { escapeValue: false },
                    })}
                  </span>
                )}
                <span className="text-xs text-neutral-500 ml-14">
                  {skill.source} / {skill.type}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex gap-6 p-6 justify-end">
        <BrandButton
          testId="skills-save-button"
          variant="primary"
          type="button"
          isDisabled={isPending || !hasChanges}
          onClick={handleSave}
        >
          {!isPending && t(I18nKey.SETTINGS$SAVE_CHANGES)}
          {isPending && t(I18nKey.SETTINGS$SAVING)}
        </BrandButton>
      </div>
    </div>
  );
}

export default SkillsSettingsScreen;
