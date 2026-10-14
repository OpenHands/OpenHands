import React from "react";
import { ActiveProfileScopeNotice } from "#/components/features/settings/agent-profiles/active-profile-scope-notice";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useActiveProfileBehavior } from "#/hooks/use-active-profile-behavior";
import type { SettingsValue } from "#/types/settings";
import { SdkSectionPage, type SettingsSourceConfig } from "./sdk-section-page";

type ProfileSection = "condenser" | "verification";

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

/**
 * An agent-settings section that lives on the active agent profile locally
 * and in the global settings on Cloud. Keys in `globalKeys`, and every
 * conversation setting, stay global either way.
 */
export function ActiveProfileSectionPage({
  settingsSources,
  section,
  globalKeys = [],
  testId,
}: {
  settingsSources: SettingsSourceConfig[];
  section: ProfileSection;
  globalKeys?: string[];
  testId: string;
}) {
  const behavior = useActiveProfileBehavior();
  const { data: settings } = useSettings();
  const { mutateAsync: saveSettings } = useSaveSettings();

  const stored = asRecord(behavior.profile?.[section]);
  const globalSection = asRecord(settings?.agent_settings?.[section]);
  const globalKeysSignature = globalKeys.join(",");
  const settingsOverride = React.useMemo(() => {
    if (!settings || !behavior.profile) return undefined;
    const shown = { ...stored };
    globalKeys.forEach((key) => {
      if (key in globalSection) shown[key] = globalSection[key];
    });
    return {
      ...settings,
      agent_settings: {
        ...settings.agent_settings,
        [section]: shown as SettingsValue,
      },
    };
  }, [settings, behavior.profile, section, globalKeysSignature]);

  if (!behavior.isLocal) {
    return <SdkSectionPage settingsSources={settingsSources} testId={testId} />;
  }
  const notice = (
    <ActiveProfileScopeNotice
      profileName={behavior.profileName}
      isAcp={behavior.isAcp}
    />
  );
  if (behavior.isAcp) return <div data-testid={testId}>{notice}</div>;
  if (!settingsOverride) return <LlmSettingsInputsSkeleton />;

  const handleSave = async (payload: Record<string, unknown>) => {
    const {
      conversation_settings_diff: conversationDiff,
      agent_settings_diff,
    } = payload as {
      conversation_settings_diff?: Record<string, unknown>;
      agent_settings_diff?: Record<string, unknown>;
    };
    const sectionDiff = asRecord(agent_settings_diff?.[section]);
    const globalDiff = Object.fromEntries(
      Object.entries(sectionDiff).filter(([key]) => globalKeys.includes(key)),
    );
    const profileDiff = Object.fromEntries(
      Object.entries(sectionDiff).filter(([key]) => !globalKeys.includes(key)),
    );

    const globalPayload = {
      ...(conversationDiff
        ? { conversation_settings_diff: conversationDiff }
        : {}),
      ...(Object.keys(globalDiff).length > 0
        ? { agent_settings_diff: { [section]: globalDiff } }
        : {}),
    };
    await Promise.all([
      Object.keys(globalPayload).length > 0
        ? saveSettings(globalPayload)
        : undefined,
      Object.keys(profileDiff).length > 0
        ? behavior.save({
            [section]: { ...stored, ...profileDiff },
          })
        : undefined,
    ]);
  };

  return (
    <SdkSectionPage
      key={behavior.profileName}
      settingsSources={settingsSources}
      settingsOverride={settingsOverride}
      onSave={handleSave}
      header={() => notice}
      testId={testId}
    />
  );
}
