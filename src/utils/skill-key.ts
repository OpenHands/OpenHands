import type { SkillInfo } from "#/types/settings";

/**
 * React key for one skills-list entry. A local skill can share its name with
 * the bundled catalog entry (or with another local skill at a different
 * path), so the name alone is not unique: duplicate keys break reconciliation
 * and leave stale or repeated cards behind when the list changes. The source
 * path - or "public" for bundled entries - is what tells two same-name
 * definitions apart.
 */
export function getSkillKey(skill: SkillInfo): string {
  const source = skill.source?.trim();
  return source ? `${source}#${skill.name}` : skill.name;
}
