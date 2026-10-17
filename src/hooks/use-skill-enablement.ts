import React from "react";
import { useTranslation } from "react-i18next";
import { useActiveProfileBehavior } from "#/hooks/use-active-profile-behavior";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  buildSkillEnablementFilter,
  CATALOG_SKILL_NAMES,
  isCatalogSkill,
  profileSkillEnablement,
  resolveEnabledCatalogSkills,
  type SkillEnablement,
} from "#/utils/skill-enablement";

/**
 * Where the selection lives: a local agent profile, the active one unless
 * `agentProfileId` names another, and the account's settings on Cloud, which
 * creates conversations from its own catalog and keeps that catalog deny-list
 * governed.
 */
function useEnablementSource(agentProfileId?: string | null): {
  enablement: SkillEnablement | undefined;
  /** The local profile the selection belongs to; null on Cloud. */
  profileName: string | null;
  /** The local profile runs an ACP agent, which has no selection. */
  isAcp: boolean;
  isError: boolean;
  retry: () => void;
  save: (enablement: SkillEnablement) => Promise<unknown>;
} {
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { mutateAsync: saveSettings } = useSaveSettings();
  const active = useActiveProfileBehavior(agentProfileId);
  const cloudDisabled = settings?.disabled_skills;
  const publicSkills = active.profile?.public_skills;
  const profileDisabled = active.profile?.disabled_skills;
  const hasProfile = active.profile !== null;
  const settingsReady = !settingsLoading && !!settings;

  const enablement = React.useMemo<SkillEnablement | undefined>(() => {
    if (!active.isLocal) {
      return !settingsReady
        ? undefined
        : {
            enabledSkills: [...CATALOG_SKILL_NAMES],
            disabledSkills: cloudDisabled ?? [],
          };
    }
    return hasProfile
      ? profileSkillEnablement({
          public_skills: publicSkills,
          disabled_skills: profileDisabled,
        })
      : undefined;
  }, [
    active.isLocal,
    settingsReady,
    cloudDisabled,
    hasProfile,
    publicSkills,
    profileDisabled,
  ]);

  return {
    enablement,
    profileName: active.isLocal ? active.profileName : null,
    isAcp: active.isLocal && active.isAcp,
    isError: active.isError,
    retry: active.retry,
    save: active.isLocal
      ? (next) =>
          active.save({
            public_skills: resolveEnabledCatalogSkills(next),
            disabled_skills: next.disabledSkills ?? [],
          })
      : (next) => saveSettings({ disabled_skills: next.disabledSkills ?? [] }),
  };
}

/** Read-only view of the rule, for surfaces that list skills without toggling. */
export function useSkillEnabledFilter(
  agentProfileId?: string | null,
): (skill: SkillInfo) => boolean {
  const { enablement } = useEnablementSource(agentProfileId);

  return React.useMemo(() => {
    const isEnabled = buildSkillEnablementFilter(enablement ?? {});
    return (skill: SkillInfo) => isEnabled(skill.name);
  }, [enablement]);
}

export interface SkillEnablementController {
  isEnabled: (skill: SkillInfo) => boolean;
  setEnabled: (skillName: string, enabled: boolean) => void;
  /** The local profile being edited; null on Cloud. */
  profileName: string | null;
  /** The local profile runs an ACP agent, so nothing can be toggled. */
  isAcp: boolean;
  /** The local profile could not be loaded, so nothing can be toggled. */
  isError: boolean;
  retry: () => void;
}

/** Stable form of both lists, so an unchanged set can skip the save. */
function snapshot(enablement: SkillEnablement): string {
  return JSON.stringify([
    [...resolveEnabledCatalogSkills(enablement)].sort(),
    [...(enablement.disabledSkills ?? [])].sort(),
  ]);
}

/** Same array reference when the membership already matches, to skip a save. */
function withMembership(
  list: string[],
  name: string,
  present: boolean,
): string[] {
  if (list.includes(name) === present) return list;
  return present ? [...list, name] : list.filter((entry) => entry !== name);
}

/**
 * Shared toggle state for every surface that switches skills on and off.
 *
 * Which of the two lists a skill belongs to is decided here alone, so one
 * surface cannot write a preference another cannot see.
 */
export function useSkillEnablement(
  agentProfileId?: string | null,
): SkillEnablementController {
  const { t } = useTranslation("openhands");
  const {
    enablement: source,
    profileName,
    isAcp,
    isError,
    retry,
    save,
  } = useEnablementSource(agentProfileId);
  const usesCatalogAllowList = profileName !== null;

  const [enablement, setEnablement] = React.useState<SkillEnablement>({});
  // The latest selection, read by toggles so a save never sees stale state.
  const enablementRef = React.useRef<SkillEnablement | null>(null);
  const saveRef = React.useRef(save);
  saveRef.current = save;

  React.useEffect(() => {
    if (!source) return;
    enablementRef.current = source;
    setEnablement(source);
  }, [profileName, source]);

  const isEnabled = React.useMemo(() => {
    const enabled = buildSkillEnablementFilter(enablement);
    return (skill: SkillInfo) => enabled(skill.name);
  }, [enablement]);

  const setEnabled = React.useCallback(
    (skillName: string, enabled: boolean) => {
      if (isAcp) {
        displayErrorToast(
          t(I18nKey.SETTINGS$BEHAVIOR_ACP_PROFILE, { name: profileName }),
        );
        return;
      }
      const previous = enablementRef.current;
      // Nothing to edit until the saved selection has loaded.
      if (!previous) {
        if (isError) {
          displayErrorToast(t(I18nKey.SETTINGS$BEHAVIOR_PROFILE_LOAD_FAILED));
        }
        return;
      }
      const allowListed = usesCatalogAllowList && isCatalogSkill(skillName);
      const next: SkillEnablement = {
        enabledSkills: allowListed
          ? withMembership(
              resolveEnabledCatalogSkills(previous),
              skillName,
              enabled,
            )
          : previous.enabledSkills,
        // A catalog skill switched back on also leaves the deny-list, where
        // a profile's earlier deny-list model may still hold it.
        disabledSkills: withMembership(
          previous.disabledSkills ?? [],
          skillName,
          !enabled && !allowListed,
        ),
      };
      if (snapshot(next) === snapshot(previous)) return;
      enablementRef.current = next;
      setEnablement(next);
      saveRef.current(next).catch((error) => {
        displayErrorToast(
          retrieveAxiosErrorMessage(error) ||
            (error instanceof Error ? error.message : "") ||
            t(I18nKey.ERROR$GENERIC),
        );
      });
    },
    [usesCatalogAllowList, isAcp, isError, profileName, t],
  );

  return { isEnabled, setEnabled, profileName, isAcp, isError, retry };
}
