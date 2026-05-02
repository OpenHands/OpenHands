import React from "react";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useSkills } from "#/hooks/query/use-skills";
import {
  useSetPersonalSkillsRepo,
  useUpdatePersonalSkillsRepo,
  useRemovePersonalSkillsRepo,
} from "#/hooks/mutation/use-personal-skills-repo";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

function formatSkillName(name: string): string {
  return name
    .replace(/[-_]/g, " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function SkillsSettingsScreen() {
  const { t } = useTranslation();

  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: skills, isLoading: skillsLoading } = useSkills();

  const setRepo = useSetPersonalSkillsRepo();
  const updateRepo = useUpdatePersonalSkillsRepo();
  const removeRepo = useRemovePersonalSkillsRepo();

  const [disabledSet, setDisabledSet] = React.useState<Set<string>>(new Set());
  const [hasChanges, setHasChanges] = React.useState(false);
  const [repoUrl, setRepoUrl] = React.useState("");

  React.useEffect(() => {
    if (settings?.disabled_skills) {
      setDisabledSet(new Set(settings.disabled_skills));
    }
  }, [settings?.disabled_skills]);

  React.useEffect(() => {
    if (settings?.personal_skills_repo_url) {
      setRepoUrl(settings.personal_skills_repo_url);
    }
  }, [settings?.personal_skills_repo_url]);

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
          displayErrorToast(
            retrieveAxiosErrorMessage(error as AxiosError) ||
              t(I18nKey.ERROR$GENERIC),
          );
        },
      },
    );
  };

  const handleSetRepo = () => {
    if (!repoUrl.trim()) return;
    setRepo.mutate(repoUrl.trim(), {
      onSuccess: () => displaySuccessToast(t(I18nKey.SETTINGS$SAVED)),
      onError: (error) =>
        displayErrorToast(
          retrieveAxiosErrorMessage(error as AxiosError) ||
            t(I18nKey.ERROR$GENERIC),
        ),
    });
  };

  const handleUpdateRepo = () => {
    updateRepo.mutate(undefined, {
      onSuccess: () => displaySuccessToast(t(I18nKey.SETTINGS$SAVED)),
      onError: (error) =>
        displayErrorToast(
          retrieveAxiosErrorMessage(error as AxiosError) ||
            t(I18nKey.ERROR$GENERIC),
        ),
    });
  };

  const handleRemoveRepo = () => {
    removeRepo.mutate(undefined, {
      onSuccess: () => {
        setRepoUrl("");
        displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      },
      onError: (error) =>
        displayErrorToast(
          retrieveAxiosErrorMessage(error as AxiosError) ||
            t(I18nKey.ERROR$GENERIC),
        ),
    });
  };

  const isLoading = settingsLoading || skillsLoading || !settings;
  const hasRepo = !!settings?.personal_skills_repo_url;
  const repoChanged =
    repoUrl.trim() !== (settings?.personal_skills_repo_url ?? "");

  return (
    <div data-testid="skills-settings-screen" className="flex flex-col h-full">
      {/* Personal Skills Repo Section */}
      <div className="mb-6 pb-6 border-b border-[#717888]">
        <h3 className="text-sm font-semibold mb-1">
          {t(I18nKey.SETTINGS$SKILLS_REPO_TITLE)}
        </h3>
        <p className="text-xs text-tertiary-alt mb-4">
          {t(I18nKey.SETTINGS$SKILLS_REPO_DESCRIPTION)}
        </p>

        <div className="flex items-end gap-3 mb-3">
          <SettingsInput
            testId="personal-skills-repo-url"
            label={t(I18nKey.SETTINGS$SKILLS_REPO_URL_LABEL)}
            type="text"
            placeholder="https://github.com/user/my-skills.git"
            value={repoUrl}
            onChange={setRepoUrl}
            className="flex-1 max-w-none"
          />
          <BrandButton
            testId="personal-skills-repo-save"
            variant="primary"
            type="button"
            isDisabled={
              setRepo.isPending || !repoUrl.trim() || (!repoChanged && hasRepo)
            }
            onClick={handleSetRepo}
          >
            {(() => {
              if (setRepo.isPending) return t(I18nKey.SETTINGS$SAVING);
              if (hasRepo && !repoChanged) return t(I18nKey.SETTINGS$SAVED);
              return t(I18nKey.BUTTON$SAVE);
            })()}
          </BrandButton>
        </div>

        {hasRepo && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-4 text-xs text-tertiary-alt">
              <span>
                {t(I18nKey.SETTINGS$SKILLS_REPO_PINNED_COMMIT)}{" "}
                <code className="bg-tertiary px-1.5 py-0.5 rounded text-[11px] font-mono">
                  {settings.personal_skills_repo_commit?.slice(0, 12)}
                </code>
              </span>
              {settings.personal_skills_repo_updated_at && (
                <span>
                  {t(I18nKey.SETTINGS$SKILLS_REPO_UPDATED)}{" "}
                  {new Date(
                    settings.personal_skills_repo_updated_at,
                  ).toLocaleString()}
                </span>
              )}
            </div>
            <div className="flex gap-2">
              <BrandButton
                testId="personal-skills-repo-update"
                variant="secondary"
                type="button"
                isDisabled={updateRepo.isPending}
                onClick={handleUpdateRepo}
              >
                {updateRepo.isPending
                  ? t(I18nKey.SETTINGS$SAVING)
                  : t(I18nKey.SETTINGS$SKILLS_REPO_UPDATE)}
              </BrandButton>
              <BrandButton
                testId="personal-skills-repo-remove"
                variant="ghost-danger"
                type="button"
                isDisabled={removeRepo.isPending}
                onClick={handleRemoveRepo}
              >
                {removeRepo.isPending
                  ? t(I18nKey.SETTINGS$SAVING)
                  : t(I18nKey.SETTINGS$SKILLS_REPO_REMOVE)}
              </BrandButton>
            </div>
          </div>
        )}
      </div>

      {/* Skills Toggle Section */}
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
                  {formatSkillName(skill.name)}
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
