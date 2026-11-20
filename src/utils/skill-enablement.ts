import {
  DEFAULT_ENABLED_SKILL_NAMES,
  SKILLS_CATALOG,
} from "@openhands/extensions/skills";
import type { SkillInfo } from "#/types/settings";

/** Every skill bundled from `@openhands/extensions`, in catalog order. */
export const CATALOG_SKILL_NAMES: readonly string[] = SKILLS_CATALOG.map(
  (entry) => entry.name,
);

const CATALOG_SKILL_NAME_SET = new Set(CATALOG_SKILL_NAMES);
const RECOMMENDED_SKILL_NAME_SET = new Set(DEFAULT_ENABLED_SKILL_NAMES);

export function isCatalogSkill(name: string): boolean {
  return CATALOG_SKILL_NAME_SET.has(name);
}

/**
 * Whether a skill belongs to the public catalog, as the server reports it.
 * Without a server listing, the bundled catalog stands in.
 */
export function publicSkillTest(
  catalog: readonly SkillInfo[] | undefined,
): (name: string) => boolean {
  if (!catalog) return isCatalogSkill;
  const publicNames = new Set(
    catalog
      .filter((skill) => skill.origin === "public")
      .map((skill) => skill.name),
  );
  return (name) => publicNames.has(name);
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
  isPublic: (name: string) => boolean = isCatalogSkill,
): (skillName: string) => boolean {
  const enabled = new Set(resolveEnabledCatalogSkills(enablement));
  const disabled = new Set(enablement.disabledSkills ?? []);

  return (skillName) => {
    if (disabled.has(skillName)) return false;
    return !isPublic(skillName) || enabled.has(skillName);
  };
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
 * Switch one skill on or off. A catalog skill moves in the allow-list, any
 * other skill in the deny-list.
 */
export function toggleSkill(
  enablement: SkillEnablement,
  skillName: string,
  enabled: boolean,
  isPublic: (name: string) => boolean = isCatalogSkill,
): SkillEnablement {
  const allowListed = isPublic(skillName);
  return {
    enabledSkills: allowListed
      ? withMembership(
          resolveEnabledCatalogSkills(enablement),
          skillName,
          enabled,
        )
      : enablement.enabledSkills,
    // A catalog skill switched back on also leaves the deny-list, where an
    // older selection may still hold it.
    disabledSkills: withMembership(
      enablement.disabledSkills ?? [],
      skillName,
      !enabled && !allowListed,
    ),
  };
}

/** The profile fields that store a selection. */
export function toProfileSkillFields(enablement: SkillEnablement): {
  public_skills: string[];
  disabled_skills: string[];
} {
  return {
    public_skills: resolveEnabledCatalogSkills(enablement),
    disabled_skills: enablement.disabledSkills ?? [],
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

/** A profile's skill selection; a profile without `public_skills` loads the whole catalog. */
export function profileSkillEnablement(
  profile: {
    public_skills?: string[] | null;
    disabled_skills?: string[];
  },
  publicNames: readonly string[] = CATALOG_SKILL_NAMES,
): SkillEnablement {
  return {
    enabledSkills: profile.public_skills ?? [...publicNames],
    disabledSkills: profile.disabled_skills ?? [],
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

/** The default list new agent profiles start from, read as a pre-allow-list workspace migrates. */
export function defaultSkillEnablement(settings: {
  enabled_skills?: string[];
  disabled_skills?: string[];
}): SkillEnablement {
  const stored = toSkillEnablement(settings);
  const migrated = migrateSkillEnablement(stored);
  return migrated
    ? {
        enabledSkills: migrated.enabled_skills,
        disabledSkills: migrated.disabled_skills,
      }
    : stored;
}

/** Whether two selections store the same lists, ignoring order. */
export function sameSkillSelection(
  a: SkillEnablement,
  b: SkillEnablement,
): boolean {
  const key = (enablement: SkillEnablement) =>
    JSON.stringify([
      [...resolveEnabledCatalogSkills(enablement)].sort(),
      [...(enablement.disabledSkills ?? [])].sort(),
    ]);
  return key(a) === key(b);
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
