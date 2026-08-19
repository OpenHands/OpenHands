import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AgentDefinitionLevel } from "#/types/settings";

interface AgentLevelBadgeProps {
  level: AgentDefinitionLevel;
}

/** Theme-aware pill chrome for agent level badges only (not shared metadata pills). */
const AGENT_LEVEL_BADGE_CLASS_NAME =
  "inline-flex shrink-0 items-center whitespace-nowrap rounded-full border border-text-secondary/35 bg-text-secondary/12 px-2 py-0.5 text-[11px] font-medium leading-4 text-tertiary-light";

const LEVEL_CONFIG: Record<AgentDefinitionLevel, { labelKey: I18nKey }> = {
  builtin: { labelKey: I18nKey.SETTINGS$AGENTS_LEVEL_BUILTIN },
  project: { labelKey: I18nKey.SETTINGS$AGENTS_LEVEL_PROJECT },
  user: { labelKey: I18nKey.SETTINGS$AGENTS_LEVEL_USER },
  plugin: { labelKey: I18nKey.SETTINGS$AGENTS_LEVEL_PLUGIN },
  programmatic: { labelKey: I18nKey.SETTINGS$AGENTS_LEVEL_PROGRAMMATIC },
};

export function getAgentLevelLabelKey(level: AgentDefinitionLevel): I18nKey {
  return LEVEL_CONFIG[level].labelKey;
}

export function AgentLevelBadge({ level }: AgentLevelBadgeProps) {
  const { t } = useTranslation("openhands");
  const config = LEVEL_CONFIG[level];
  return (
    <span
      data-testid={`agent-level-badge-${level}`}
      className={AGENT_LEVEL_BADGE_CLASS_NAME}
    >
      {t(config.labelKey)}
    </span>
  );
}
