import React from "react";
import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  isCatalogSkill,
  resolveEnabledCatalogSkills,
  toSkillEnablement,
} from "#/utils/skill-enablement";

export interface SkillEnablementController {
  /** Whether a skill will be loaded into new conversations. */
  isEnabled: (skill: SkillInfo) => boolean;
  setEnabled: (skillName: string, enabled: boolean) => void;
}

/**
 * Shared toggle state for every surface that switches skills on and off.
 *
 * The two lists it writes are not symmetric — see `SkillEnablement` — so both
 * the Customize page and the conversation drawer have to agree on which list a
 * given skill belongs to. Keeping that decision here is what stops one surface
 * from writing a preference the other cannot see.
 *
 * Cloud backends stay on the plain deny-list: cloud creates conversations from
 * its own server-side catalog and never reads `enabled_skills`, so showing
 * most of the catalog switched off there would misreport what the agent loads.
 */
export function useSkillEnablement(): SkillEnablementController {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const usesCatalogAllowList = backend.kind !== "cloud";
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { mutate: saveSettings } = useSaveSettings();

  const [enabledCatalogSet, setEnabledCatalogSet] = React.useState<Set<string>>(
    () => new Set(),
  );
  const [disabledSet, setDisabledSet] = React.useState<Set<string>>(
    () => new Set(),
  );
  const [hasHydratedInitialSettings, setHasHydratedInitialSettings] =
    React.useState(false);

  React.useEffect(() => {
    if (settingsLoading || !settings) return;
    setEnabledCatalogSet(
      new Set(resolveEnabledCatalogSkills(toSkillEnablement(settings))),
    );
    setDisabledSet(new Set(settings.disabled_skills ?? []));
    setHasHydratedInitialSettings(true);
  }, [settingsLoading, settings?.enabled_skills, settings?.disabled_skills]);

  React.useEffect(() => {
    if (!hasHydratedInitialSettings) return;
    saveSettings(
      usesCatalogAllowList
        ? {
            enabled_skills: Array.from(enabledCatalogSet),
            disabled_skills: Array.from(disabledSet),
          }
        : { disabled_skills: Array.from(disabledSet) },
      {
        onError: (error) => {
          displayErrorToast(
            retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
          );
        },
      },
    );
  }, [
    enabledCatalogSet,
    disabledSet,
    hasHydratedInitialSettings,
    usesCatalogAllowList,
    saveSettings,
    t,
  ]);

  const isEnabled = React.useCallback(
    (skill: SkillInfo) => {
      if (disabledSet.has(skill.name)) return false;
      if (!usesCatalogAllowList || !isCatalogSkill(skill.name)) return true;
      return enabledCatalogSet.has(skill.name);
    },
    [disabledSet, enabledCatalogSet, usesCatalogAllowList],
  );

  const setEnabled = React.useCallback(
    (skillName: string, enabled: boolean) => {
      if (usesCatalogAllowList && isCatalogSkill(skillName)) {
        setEnabledCatalogSet((previous) => {
          const next = new Set(previous);
          if (enabled) {
            next.add(skillName);
          } else {
            next.delete(skillName);
          }
          return next;
        });
      }

      // A catalog skill switched back on also has to leave the deny-list: an
      // unmigrated workspace can still hold its name there, and that entry
      // would otherwise veto the allow-list.
      setDisabledSet((previous) => {
        const shouldDeny =
          !enabled && !(usesCatalogAllowList && isCatalogSkill(skillName));
        if (shouldDeny === previous.has(skillName)) return previous;
        const next = new Set(previous);
        if (shouldDeny) {
          next.add(skillName);
        } else {
          next.delete(skillName);
        }
        return next;
      });
    },
    [usesCatalogAllowList],
  );

  return { isEnabled, setEnabled };
}
