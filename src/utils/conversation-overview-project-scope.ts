import type { Automation } from "#/types/automation";
import type { SkillInfo } from "#/types/settings";
import {
  getSkillScope,
  SKILL_SCOPE_ORDER,
  type SkillScope,
} from "#/utils/skill-scope";

export const CONVERSATION_OVERVIEW_PROJECT_SCOPE = {
  project: "project",
  all: "all",
} as const;

export type ConversationOverviewProjectScope =
  (typeof CONVERSATION_OVERVIEW_PROJECT_SCOPE)[keyof typeof CONVERSATION_OVERVIEW_PROJECT_SCOPE];

export function normalizeRepositoryName(
  repository: string | null | undefined,
): string | null {
  if (!repository?.trim()) {
    return null;
  }
  return repository
    .trim()
    .toLowerCase()
    .replace(/\.git$/, "");
}

export function isAutomationForRepository(
  automation: Pick<Automation, "repository">,
  repository: string | null | undefined,
): boolean {
  const target = normalizeRepositoryName(repository);
  const automationRepo = normalizeRepositoryName(automation.repository);
  if (!target || !automationRepo) {
    return false;
  }
  return automationRepo === target;
}

export function isSkillForProject(
  skill: SkillInfo,
  projectDir?: string | null,
): boolean {
  return getSkillScope(skill, projectDir) === "project";
}

export function filterAutomationsByProjectScope(
  automations: Automation[],
  scope: ConversationOverviewProjectScope,
  repository: string | null | undefined,
): Automation[] {
  if (scope === CONVERSATION_OVERVIEW_PROJECT_SCOPE.all) {
    return automations;
  }
  return automations.filter((automation) =>
    isAutomationForRepository(automation, repository),
  );
}

export function filterSkillsByProjectScope(
  skills: SkillInfo[],
  scope: ConversationOverviewProjectScope,
  projectDir?: string | null,
): SkillInfo[] {
  if (scope === CONVERSATION_OVERVIEW_PROJECT_SCOPE.all) {
    return skills;
  }
  return skills.filter((skill) => isSkillForProject(skill, projectDir));
}

export function sortAutomationsByProjectRelevance(
  automations: Automation[],
  repository: string | null | undefined,
): Automation[] {
  return [...automations].sort((left, right) => {
    const leftMatch = isAutomationForRepository(left, repository) ? 0 : 1;
    const rightMatch = isAutomationForRepository(right, repository) ? 0 : 1;
    if (leftMatch !== rightMatch) {
      return leftMatch - rightMatch;
    }
    return left.name.localeCompare(right.name);
  });
}

export function sortSkillsByProjectRelevance(
  skills: SkillInfo[],
  projectDir?: string | null,
): SkillInfo[] {
  return [...skills].sort((left, right) => {
    const leftScope = getSkillScope(left, projectDir);
    const rightScope = getSkillScope(right, projectDir);
    const leftOrder = SKILL_SCOPE_ORDER.indexOf(leftScope as SkillScope);
    const rightOrder = SKILL_SCOPE_ORDER.indexOf(rightScope as SkillScope);
    if (leftOrder !== rightOrder) {
      return leftOrder - rightOrder;
    }
    return left.name.localeCompare(right.name);
  });
}

export function countAutomationsForRepository(
  automations: readonly Automation[],
  repository: string | null | undefined,
): number {
  return automations.filter((automation) =>
    isAutomationForRepository(automation, repository),
  ).length;
}

export function countSkillsForProject(
  skills: readonly SkillInfo[],
  projectDir?: string | null,
): number {
  return skills.filter((skill) => isSkillForProject(skill, projectDir)).length;
}
