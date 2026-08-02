import type { Microagent } from "#/api/open-hands.types";
import type { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";

export type SlashCommandSkill = SkillInfo | Microagent;

export interface SlashCommandItem {
  skill: SlashCommandSkill;
  /** The slash command string, e.g. "/random-number" */
  command: string;
}

export type SlashCommandAvailability =
  | "always"
  | "conversation"
  | "confirmation"
  | "manual-condensation-conversation"
  | "local-conversation";

export interface BuiltInSlashCommandItem extends SlashCommandItem {
  availability: SlashCommandAvailability;
  /** Frontend-owned display copy; dynamic skills continue to use their data. */
  descriptionKey: I18nKey;
}

export interface LoadedSkillResource {
  name: string;
  description?: string | null;
  source?: string | null;
}

export interface LoadedHookResource {
  hookType: string;
  commands: string[];
}

export interface LoadedMcpResource {
  name: string;
  transport?: string | null;
}

export interface LoadedResources {
  skills: LoadedSkillResource[];
  /** Null when loaded hooks are unsupported or their request failed. */
  hooks: LoadedHookResource[] | null;
  /** Null when the active backend cannot report loaded MCPs. */
  mcps: LoadedMcpResource[] | null;
}
