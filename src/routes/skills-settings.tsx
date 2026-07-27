import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useSkills } from "#/hooks/query/use-skills";
import { ExtensionsNavigation } from "#/components/features/skills/extensions-navigation";
import { SkillCard } from "#/components/features/skills/skill-card";
import { SkillDetailModal } from "#/components/features/skills/skill-detail-modal";
import { AddSkillModal } from "#/components/features/skills/add-skill-modal";
import { SkillsToolbar } from "#/components/features/skills/skills-toolbar";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { SkillTypeFilter } from "#/components/features/skills/skill-type-filter";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { cn } from "#/utils/utils";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import type { SkillInfo } from "#/types/settings";
import { getSkillCardDescription } from "#/components/features/skills/get-skill-card-description";

function matchesSearch(skill: SkillInfo, query: string): boolean {
  if (!query) return true;
  const haystacks = [
    skill.name,
    getSkillCardDescription(skill),
    skill.description ?? "",
    skill.content ?? "",
    skill.license ?? "",
    skill.compatibility ?? "",
    ...(skill.triggers ?? []),
    ...(skill.allowed_tools ?? []),
  ];
  const lowered = query.toLowerCase();
  return haystacks.some((value) => value.toLowerCase().includes(lowered));
}

function SkillsSettingsScreen() {
  const { t } = useTranslation("openhands");

  const { mutate: saveSettings } = useSaveSettings();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: skills, isLoading: skillsLoading } = useSkills();

  const [disabledSet, setDisabledSet] = React.useState<Set<string>>(new Set());
  const [hasHydratedInitialSettings, setHasHydratedInitialSettings] =
    React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [typeFilter, setTypeFilter] = React.useState<SkillTypeFilter>("all");
  const [selectedSkill, setSelectedSkill] = React.useState<SkillInfo | null>(
    null,
  );
  const [showAddSkillModal, setShowAddSkillModal] = React.useState(false);

  // Sync local state with server settings when data first arrives
  React.useEffect(() => {
    if (settingsLoading || !settings) return;
    setDisabledSet(new Set(settings.disabled_skills ?? []));
    setHasHydratedInitialSettings(true);
  }, [settingsLoading, settings?.disabled_skills]);

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
  };

  // Auto-save skill toggles once initial settings are loaded.
  React.useEffect(() => {
    if (!hasHydratedInitialSettings) return;
    saveSettings(
      { disabled_skills: Array.from(disabledSet) },
      {
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
      },
    );
  }, [disabledSet, hasHydratedInitialSettings, saveSettings, t]);

  const isLoading = settingsLoading || skillsLoading || !settings;

  const filteredSkills = React.useMemo(() => {
    if (!skills) return [];
    return skills.filter(
      (skill) =>
        (typeFilter === "all" || skill.type === typeFilter) &&
        matchesSearch(skill, searchQuery),
    );
  }, [skills, typeFilter, searchQuery]);

  return (
    <div
      data-testid="skills-settings-screen"
      className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <h2 className="text-xl font-semibold leading-6 text-foreground">
                {t(I18nKey.SETTINGS$SKILLS_TITLE)}
              </h2>
              <div
                data-testid="skills-settings-description"
                className="max-w-2xl text-sm text-tertiary-light"
              >
                {t(I18nKey.SETTINGS$SKILLS_PAGE_DESCRIPTION)}
              </div>
            </div>
            <BrandButton
              type="button"
              variant="secondary"
              testId="skills-add-skill-button"
              className="flex-shrink-0 whitespace-nowrap"
              onClick={() => setShowAddSkillModal(true)}
            >
              {t(I18nKey.SETTINGS$SKILLS_ADD_BUTTON)}
            </BrandButton>
          </div>

          {isLoading && (
            <div className="flex flex-col gap-4">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-24 rounded-2xl bg-tertiary animate-pulse"
                />
              ))}
            </div>
          )}

          {!isLoading && (!skills || skills.length === 0) && (
            <div
              data-testid="skills-empty"
              className={extensionModuleEmptyStateClassName}
            >
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.SETTINGS$SKILLS_NO_SKILLS)}
              </p>
            </div>
          )}

          {!isLoading && skills && skills.length > 0 && (
            <>
              <SkillsToolbar
                search={searchQuery}
                onSearchChange={setSearchQuery}
                typeFilter={typeFilter}
                onTypeFilterChange={setTypeFilter}
              />
              {filteredSkills.length === 0 ? (
                <div
                  data-testid="skills-no-match"
                  className={extensionModuleEmptyStateClassName}
                >
                  <p className="text-sm text-tertiary-light">
                    {t(I18nKey.SETTINGS$SKILLS_NO_MATCH)}
                  </p>
                </div>
              ) : (
                <section
                  className={cn(
                    "flex min-w-0 flex-col gap-3",
                    extensionModuleCardGridContainerClassName,
                  )}
                >
                  <div className={extensionModuleCardGridClassName}>
                    {filteredSkills.map((skill) => (
                      <SkillCard
                        key={skill.name}
                        skill={skill}
                        enabled={!disabledSet.has(skill.name)}
                        onOpen={() => setSelectedSkill(skill)}
                        onToggle={(enabled) =>
                          handleToggle(skill.name, enabled)
                        }
                      />
                    ))}
                  </div>
                </section>
              )}
            </>
          )}

          {selectedSkill && (
            <SkillDetailModal
              skill={selectedSkill}
              enabled={!disabledSet.has(selectedSkill.name)}
              onToggle={(enabled) => handleToggle(selectedSkill.name, enabled)}
              onClose={() => setSelectedSkill(null)}
            />
          )}

          {showAddSkillModal && (
            <AddSkillModal onClose={() => setShowAddSkillModal(false)} />
          )}
        </div>
      </main>
    </div>
  );
}

export default SkillsSettingsScreen;
