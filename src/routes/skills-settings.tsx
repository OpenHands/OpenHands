import React from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router";
import { ExtensionsNavigation } from "#/components/features/skills/extensions-navigation";
import { AddSkillModal } from "#/components/features/skills/add-skill-modal";
import { SkillDetailModal } from "#/components/features/skills/skill-detail-modal";
import {
  parseSkillFilterState,
  SKILL_FILTER_QUERY_PARAM,
  toSkillFilterSearchParams,
  type SkillFilterState,
} from "#/components/features/skills/skill-filter";
import { SkillsPage } from "#/components/features/skills/skills-page";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useSkills } from "#/hooks/query/use-skills";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import { cn } from "#/utils/utils";

/**
 * Typing must not write a history entry per keystroke, so the input keeps its
 * own state and the URL is only a debounced mirror. Facet clicks push instead,
 * because each is a discrete choice the back button should undo.
 */
const SEARCH_URL_SYNC_DELAY_MS = 300;

function SkillsSettingsScreen() {
  const { t } = useTranslation("openhands");

  const { mutate: saveSettings } = useSaveSettings();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: skills, isLoading: skillsLoading } = useSkills();

  const [searchParams, setSearchParams] = useSearchParams();
  const [queryInput, setQueryInput] = React.useState(
    () => searchParams.get(SKILL_FILTER_QUERY_PARAM) ?? "",
  );

  const [disabledSet, setDisabledSet] = React.useState<Set<string>>(new Set());
  const [hasHydratedInitialSettings, setHasHydratedInitialSettings] =
    React.useState(false);
  const [selectedSkill, setSelectedSkill] = React.useState<SkillInfo | null>(
    null,
  );
  const [showAddSkillModal, setShowAddSkillModal] = React.useState(false);

  // `query` always comes from `queryInput`, never from `searchParams`, even
  // though the URL also carries a `q` value: the input is the source of
  // truth for what's rendered, and the URL is only a debounced mirror of it
  // (see the sync effect below). Reading `q` back out of `searchParams` here
  // would race the debounce and could bounce the input's own value back at
  // it before the timeout fires.
  const filter = React.useMemo<SkillFilterState>(
    () => ({ ...parseSkillFilterState(searchParams), query: queryInput }),
    [searchParams, queryInput],
  );

  // Sync local state with server settings when data first arrives
  React.useEffect(() => {
    if (settingsLoading || !settings) return;
    setDisabledSet(new Set(settings.disabled_skills ?? []));
    setHasHydratedInitialSettings(true);
  }, [settingsLoading, settings?.disabled_skills]);

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

  React.useEffect(() => {
    const current = searchParams.get(SKILL_FILTER_QUERY_PARAM) ?? "";
    if (current === queryInput) return undefined;

    const timeout = setTimeout(() => {
      setSearchParams(
        (previous) => {
          const next = new URLSearchParams(previous);
          if (queryInput) {
            next.set(SKILL_FILTER_QUERY_PARAM, queryInput);
          } else {
            next.delete(SKILL_FILTER_QUERY_PARAM);
          }
          return next;
        },
        { replace: true },
      );
    }, SEARCH_URL_SYNC_DELAY_MS);

    return () => clearTimeout(timeout);
  }, [queryInput, searchParams, setSearchParams]);

  const handleFilterChange = (next: SkillFilterState) => {
    // A query change goes through local state, not straight to the URL (see
    // the `filter` comment above and the sync effect below).
    if (next.query !== filter.query) setQueryInput(next.query);

    // Compare facets only (query aside) so a pure query change does not also
    // trigger an immediate, undebounced URL write here; a facet change
    // always writes through, even if a query change happens alongside it.
    const facetsChanged =
      toSkillFilterSearchParams({ ...next, query: "" }).toString() !==
      toSkillFilterSearchParams({ ...filter, query: "" }).toString();
    if (facetsChanged) setSearchParams(toSkillFilterSearchParams(next));
  };

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

  return (
    <div
      data-testid="skills-settings-screen"
      className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <SkillsPage
          skills={skills ?? []}
          disabledSet={disabledSet}
          filter={filter}
          isLoading={settingsLoading || skillsLoading || !settings}
          onFilterChange={handleFilterChange}
          onToggle={handleToggle}
          onOpenSkill={setSelectedSkill}
          onAddSkill={() => setShowAddSkillModal(true)}
        />

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
      </main>
    </div>
  );
}

export default SkillsSettingsScreen;
