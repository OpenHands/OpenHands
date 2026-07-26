import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";
import { SkillCard } from "./skill-card";
import { SkillFacetRail } from "./skill-facet-rail";
import { SkillFiltersModal } from "./skill-filters-modal";
import {
  applySkillFilters,
  buildSkillFacetGroups,
  clearSkillFilterFacets,
  countActiveFilters,
  toggleSkillFilterValue,
  type SkillFacetGroupId,
  type SkillFilterState,
} from "./skill-filter";
import { SkillsToolbar } from "./skills-toolbar";

interface SkillsPageProps {
  skills: SkillInfo[];
  disabledSet: Set<string>;
  filter: SkillFilterState;
  isLoading: boolean;
  onFilterChange: (next: SkillFilterState) => void;
  onToggle: (skillName: string, enabled: boolean) => void;
  onOpenSkill: (skill: SkillInfo) => void;
  onAddSkill: () => void;
}

export function SkillsPage({
  skills,
  disabledSet,
  filter,
  isLoading,
  onFilterChange,
  onToggle,
  onOpenSkill,
  onAddSkill,
}: SkillsPageProps) {
  const { t } = useTranslation("openhands");
  const [isFiltersModalOpen, setIsFiltersModalOpen] = React.useState(false);

  const groups = React.useMemo(
    () => buildSkillFacetGroups(skills, disabledSet, filter),
    [skills, disabledSet, filter],
  );

  const visibleSkills = React.useMemo(
    () => applySkillFilters(skills, disabledSet, filter),
    [skills, disabledSet, filter],
  );

  const activeFilterCount = countActiveFilters(filter);

  const handleToggleFacet = (groupId: SkillFacetGroupId, value: string) =>
    onFilterChange(toggleSkillFilterValue(filter, groupId, value));

  const handleClearFacets = () =>
    onFilterChange(clearSkillFilterFacets(filter));

  return (
    <div
      data-testid="skills-page"
      className="mx-auto flex w-full min-w-0 max-w-[1100px] flex-col gap-6"
    >
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
          onClick={onAddSkill}
        >
          {t(I18nKey.SETTINGS$SKILLS_ADD_BUTTON)}
        </BrandButton>
      </div>

      {isLoading ? (
        <div className="flex flex-col gap-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-24 rounded-2xl bg-tertiary animate-pulse"
            />
          ))}
        </div>
      ) : null}

      {!isLoading && skills.length === 0 ? (
        <div
          data-testid="skills-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-tertiary-light">
            {t(I18nKey.SETTINGS$SKILLS_NO_SKILLS)}
          </p>
        </div>
      ) : null}

      {!isLoading && skills.length > 0 ? (
        <>
          <SkillsToolbar
            search={filter.query}
            onSearchChange={(query) => onFilterChange({ ...filter, query })}
            activeFilterCount={activeFilterCount}
            onOpenFilters={() => setIsFiltersModalOpen(true)}
          />

          <div className="flex min-w-0 gap-6">
            <SkillFacetRail
              groups={groups}
              onToggle={handleToggleFacet}
              className="hidden w-[204px] shrink-0 self-start md:flex"
            />

            <section className="flex min-w-0 flex-1 flex-col gap-3">
              <div className="flex items-center justify-between gap-3 text-xs text-tertiary-light">
                <span data-testid="skills-result-summary">
                  {t(I18nKey.SETTINGS$SKILLS_RESULT_COUNT, {
                    count: visibleSkills.length,
                  })}
                </span>
                {activeFilterCount > 0 ? (
                  <button
                    type="button"
                    data-testid="skills-clear-filters"
                    onClick={handleClearFacets}
                    className="cursor-pointer underline hover:text-white"
                  >
                    {t(I18nKey.SETTINGS$SKILLS_CLEAR_FILTERS)}
                  </button>
                ) : null}
              </div>

              {visibleSkills.length === 0 ? (
                <div
                  data-testid="skills-no-match"
                  className={extensionModuleEmptyStateClassName}
                >
                  <p className="text-sm text-tertiary-light">
                    {t(I18nKey.SETTINGS$SKILLS_NO_MATCH)}
                  </p>
                </div>
              ) : (
                <div className={cn(extensionModuleCardGridContainerClassName)}>
                  <div className={extensionModuleCardGridClassName}>
                    {visibleSkills.map((skill) => (
                      <SkillCard
                        key={skill.name}
                        skill={skill}
                        enabled={!disabledSet.has(skill.name)}
                        onOpen={() => onOpenSkill(skill)}
                        onToggle={(enabled) => onToggle(skill.name, enabled)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </section>
          </div>

          {isFiltersModalOpen ? (
            <SkillFiltersModal
              groups={groups}
              activeCount={activeFilterCount}
              onToggle={handleToggleFacet}
              onClearAll={handleClearFacets}
              onClose={() => setIsFiltersModalOpen(false)}
            />
          ) : null}
        </>
      ) : null}
    </div>
  );
}
