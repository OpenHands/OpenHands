import type { I18nKey } from "#/i18n/declaration";
import type { Microagent } from "#/api/open-hands.types";
import type { SkillInfo } from "#/types/settings";

export type SlashCommandSkill = SkillInfo | Microagent;

export interface SlashCommandItem {
  skill: SlashCommandSkill;
  /** The slash command string, e.g. "/random-number". */
  command: string;
  /** Localized description for built-ins; skill descriptions come from data. */
  descriptionKey?: I18nKey;
}
