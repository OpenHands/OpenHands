import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { SkillType } from "#/types/settings";

interface SkillTypeBadgeProps {
  type: SkillType;
}

/** Theme-aware pill chrome for skill type badges only (not shared metadata pills). */
const SKILL_TYPE_BADGE_CLASS_NAME =
  "inline-flex shrink-0 items-center whitespace-nowrap rounded-full border border-text-secondary/35 bg-text-secondary/12 px-2 py-0.5 text-[11px] font-medium leading-4 text-tertiary-light";

const TYPE_CONFIG: Record<SkillType, { labelKey: I18nKey }> = {
  agentskills: {
    labelKey: I18nKey.SETTINGS$SKILLS_TYPE_AGENTSKILLS,
  },
  knowledge: {
    labelKey: I18nKey.SETTINGS$SKILLS_TYPE_KNOWLEDGE,
  },
  repo: {
    labelKey: I18nKey.SETTINGS$SKILLS_TYPE_REPO,
  },
};

export function getSkillTypeLabelKey(type: SkillType): I18nKey {
  return TYPE_CONFIG[type].labelKey;
}

export function SkillTypeBadge({ type }: SkillTypeBadgeProps) {
  const { t } = useTranslation("openhands");
  const config = TYPE_CONFIG[type];
  return (
    <span
      data-testid={`skill-type-badge-${type}`}
      className={SKILL_TYPE_BADGE_CLASS_NAME}
    >
      {t(config.labelKey)}
    </span>
  );
}
