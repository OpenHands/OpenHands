import type { SkillType } from "#/types/settings";

export type SkillTypeFilter = "all" | SkillType;

export const SKILL_TYPE_FILTER_OPTIONS: SkillTypeFilter[] = [
  "all",
  "agentskills",
  "knowledge",
  "repo",
];
