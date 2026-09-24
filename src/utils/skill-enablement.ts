import {
  DEFAULT_ENABLED_SKILL_NAMES,
  SKILLS_CATALOG,
} from "@openhands/extensions/skills";

/** Every skill bundled from `@openhands/extensions`, in catalog order. */
export const CATALOG_SKILL_NAMES: readonly string[] = SKILLS_CATALOG.map(
  (entry) => entry.name,
);

const CATALOG_SKILL_NAME_SET = new Set(CATALOG_SKILL_NAMES);
const RECOMMENDED_SKILL_NAME_SET = new Set(DEFAULT_ENABLED_SKILL_NAMES);

export function isCatalogSkill(name: string): boolean {
  return CATALOG_SKILL_NAME_SET.has(name);
}

export function isRecommendedSkill(name: string): boolean {
  return RECOMMENDED_SKILL_NAME_SET.has(name);
}

/**
 * The two persisted lists. They cover different populations: `enabledSkills`
 * allow-lists the bundled catalog, whose every future addition would otherwise
 * be on for everyone (#16302), while `disabledSkills` keeps denying user- and
 * project-authored skills, which should be on the moment they appear.
 *
 * `undefined` means "never migrated" and must survive settings hydration.
 */
export interface SkillEnablement {
  enabledSkills?: string[];
  disabledSkills?: string[];
}

export function resolveEnabledCatalogSkills(
  enablement: SkillEnablement,
): string[] {
  return enablement.enabledSkills ?? [...DEFAULT_ENABLED_SKILL_NAMES];
}

/**
 * The one rule for "will this skill be loaded", resolved once per caller so
 * per-skill checks stay cheap.
 *
 * The deny-list still wins over the allow-list, which only matters before the
 * migration runs: until then a pre-existing "I turned this off" lives in the
 * deny-list alone.
 */
export function buildSkillEnablementFilter(
  enablement: SkillEnablement,
): (skillName: string) => boolean {
  const enabled = new Set(resolveEnabledCatalogSkills(enablement));
  const disabled = new Set(enablement.disabledSkills ?? []);

  return (skillName) => {
    if (disabled.has(skillName)) return false;
    return !isCatalogSkill(skillName) || enabled.has(skillName);
  };
}

/**
 * One-shot conversion from "all catalog skills on, minus a deny-list" to an
 * explicit allow-list; `undefined` once already migrated.
 *
 * A fresh workspace is migrated too, even though the resolver's fallback would
 * give it the same set: persisting an explicit list is what stops a later
 * `defaultEnabled` catalog addition from switching itself on.
 */
export function migrateSkillEnablement(
  enablement: SkillEnablement,
): { enabled_skills: string[]; disabled_skills: string[] } | undefined {
  if (enablement.enabledSkills !== undefined) return undefined;

  const disabled = new Set(enablement.disabledSkills ?? []);
  // A deny-list naming a catalog skill is the only evidence that this
  // workspace predates the allow-list; one holding local names alone says
  // nothing about the catalog.
  const isExistingWorkspace = [...disabled].some(isCatalogSkill);

  return {
    enabled_skills: isExistingWorkspace
      ? CATALOG_SKILL_NAMES.filter((name) => !disabled.has(name))
      : [...DEFAULT_ENABLED_SKILL_NAMES],
    // Catalog names move to the allow-list; a leftover deny entry would veto
    // a skill the user later switches back on.
    disabled_skills: [...disabled].filter((name) => !isCatalogSkill(name)),
  };
}

export function toSkillEnablement(settings: {
  enabled_skills?: string[];
  disabled_skills?: string[];
}): SkillEnablement {
  return {
    enabledSkills: settings.enabled_skills,
    disabledSkills: settings.disabled_skills,
  };
}

// Both forms a user can send: the commands a skill declares in its own
// `triggers` (what an automation card fills in — see `findAutomationCommand`),
// and `/<skill-name>`, which the detail modal's "Use skill" button inserts.
const CATALOG_SKILL_BY_SLASH_COMMAND = new Map(
  SKILLS_CATALOG.flatMap((entry) =>
    [
      `/${entry.name}`,
      ...(entry.triggers ?? []).filter((trigger) => trigger.startsWith("/")),
    ].map((command) => [command.toLowerCase(), entry.name] as const),
  ),
);

/**
 * The catalog skill a message invokes by name, if any.
 *
 * 18 of the catalog's 24 slash commands belong to skills that are off by
 * default, so without this an automation card would send its command with none
 * of the instructions behind it. Only the leading token counts: matching a
 * `/word` anywhere in prose would re-admit most of the catalog.
 */
export function findInvokedCatalogSkill(query?: string): string | undefined {
  const firstToken = query?.trim().split(/\s+/, 1)[0];
  if (!firstToken?.startsWith("/")) return undefined;
  return CATALOG_SKILL_BY_SLASH_COMMAND.get(firstToken.toLowerCase());
}
