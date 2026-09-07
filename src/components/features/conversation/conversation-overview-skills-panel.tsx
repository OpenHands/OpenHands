import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useSettings } from "#/hooks/query/use-settings";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useSkillEnablement } from "#/hooks/use-skill-enablement";
import { SkillCard } from "#/components/features/skills/skill-card";
import { SkillDetailModal } from "#/components/features/skills/skill-detail-modal";
import { AddSkillModal } from "#/components/features/skills/add-skill-modal";
import { SkillsToolbar } from "#/components/features/skills/skills-toolbar";
import { SkillFiltersModal } from "#/components/features/skills/skill-filters-modal";
import {
  applySkillFilters,
  buildSkillFacetGroups,
  clearSkillFilterFacets,
  countActiveFilters,
  toggleSkillFilterValue,
  EMPTY_SKILL_FILTER_STATE,
  type SkillFilterState,
} from "#/components/features/skills/skill-filter";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import type { SkillInfo } from "#/types/settings";
import { cn } from "#/utils/utils";
import {
  CONVERSATION_OVERVIEW_PROJECT_SCOPE,
  filterSkillsByProjectScope,
  sortSkillsByProjectRelevance,
  type ConversationOverviewProjectScope,
} from "#/utils/conversation-overview-project-scope";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";
import { ConversationOverviewProjectScopeToggle } from "./conversation-overview-project-scope-toggle";

interface ConversationOverviewSkillsPanelProps {
  openAdd: boolean;
}

export function ConversationOverviewSkillsPanel({
  openAdd,
}: ConversationOverviewSkillsPanelProps) {
  const { t } = useTranslation("openhands");
  const { isLoading: settingsLoading } = useSettings();
  const { data: conversation } = useActiveConversation();
  const { data: skills, isLoading: skillsLoading } = useConversationSkills();
  const { isEnabled, setEnabled } = useSkillEnablement();
  const addRequestKey =
    useConversationOverviewDrawerOptional()?.addRequestKey ?? 0;
  const projectDir = conversation?.selected_workspace ?? null;

  const [projectScope, setProjectScope] =
    useState<ConversationOverviewProjectScope>(
      CONVERSATION_OVERVIEW_PROJECT_SCOPE.project,
    );
  const [filter, setFilter] = useState<SkillFilterState>(
    EMPTY_SKILL_FILTER_STATE,
  );
  const [isFiltersModalOpen, setIsFiltersModalOpen] = useState(false);
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [showAddSkillModal, setShowAddSkillModal] = useState(false);

  useEffect(() => {
    if (!openAdd) {
      return;
    }
    setShowAddSkillModal(true);
  }, [openAdd]);

  useEffect(() => {
    if (addRequestKey === 0) {
      return;
    }
    setShowAddSkillModal(true);
  }, [addRequestKey]);

  const scopedSkills = useMemo(() => {
    if (!skills) {
      return [];
    }
    const filtered = filterSkillsByProjectScope(
      skills,
      projectScope,
      projectDir,
    );
    return projectScope === CONVERSATION_OVERVIEW_PROJECT_SCOPE.all
      ? sortSkillsByProjectRelevance(filtered, projectDir)
      : filtered;
  }, [skills, projectScope, projectDir]);

  const filteredSkills = useMemo(
    () => applySkillFilters(scopedSkills, isEnabled, filter),
    [scopedSkills, isEnabled, filter],
  );

  if (skillsLoading || settingsLoading) {
    return (
      <div
        data-testid="conversation-overview-skills-panel"
        className="text-sm text-muted"
      >
        …
      </div>
    );
  }

  if (!skills || skills.length === 0) {
    return (
      <div
        data-testid="conversation-overview-skills-panel"
        className="flex flex-col gap-4"
      >
        <div
          data-testid="conversation-overview-skills-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-tertiary-light">
            {t(I18nKey.SETTINGS$SKILLS_NO_SKILLS)}
          </p>
        </div>
        {showAddSkillModal ? (
          <AddSkillModal onClose={() => setShowAddSkillModal(false)} />
        ) : null}
      </div>
    );
  }

  return (
    <div
      data-testid="conversation-overview-skills-panel"
      className="flex flex-col gap-4"
    >
      <ConversationOverviewProjectScopeToggle
        value={projectScope}
        onChange={setProjectScope}
        testId="conversation-overview-skills-scope"
      />
      <SkillsToolbar
        search={filter.query}
        onSearchChange={(query) =>
          setFilter((previous) => ({ ...previous, query }))
        }
        activeFilterCount={countActiveFilters(filter)}
        onOpenFilters={() => setIsFiltersModalOpen(true)}
      />

      {scopedSkills.length === 0 ? (
        <div
          data-testid="conversation-overview-skills-project-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-tertiary-light">
            {t(I18nKey.CONVERSATION$OVERVIEW_SCOPE_PROJECT_EMPTY_SKILLS)}
          </p>
        </div>
      ) : filteredSkills.length === 0 ? (
        <div
          data-testid="conversation-overview-skills-no-match"
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
          <div className={cn(extensionModuleCardGridClassName, "grid-cols-1")}>
            {filteredSkills.map((skill) => (
              <SkillCard
                key={skill.name}
                skill={skill}
                enabled={isEnabled(skill)}
                onOpen={() => setSelectedSkill(skill)}
                onToggle={(enabled) => setEnabled(skill.name, enabled)}
              />
            ))}
          </div>
        </section>
      )}

      {selectedSkill ? (
        <SkillDetailModal
          skill={selectedSkill}
          enabled={isEnabled(selectedSkill)}
          onToggle={(enabled) => setEnabled(selectedSkill.name, enabled)}
          onClose={() => setSelectedSkill(null)}
        />
      ) : null}

      {showAddSkillModal ? (
        <AddSkillModal onClose={() => setShowAddSkillModal(false)} />
      ) : null}

      {isFiltersModalOpen ? (
        <SkillFiltersModal
          groups={buildSkillFacetGroups(scopedSkills, isEnabled, filter)}
          activeCount={countActiveFilters(filter)}
          onToggle={(groupId, value) =>
            setFilter((previous) =>
              toggleSkillFilterValue(previous, groupId, value),
            )
          }
          onClearAll={() =>
            setFilter((previous) => clearSkillFilterFacets(previous))
          }
          onClose={() => setIsFiltersModalOpen(false)}
        />
      ) : null}
    </div>
  );
}
