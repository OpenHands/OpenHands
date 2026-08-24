import {
  DEFAULT_ENABLED_SKILL_NAMES,
  SKILLS_CATALOG,
} from "@openhands/extensions/skills";

/**
 * Every skill name in the bundled `@openhands/extensions` catalog.
 *
 * Membership is what routes a skill to the allow-list rather than the
 * deny-list, so it has to match what `buildBundledSkills()` actually ships:
 * both read the same build-time snapshot.
 */
export const CATALOG_SKILL_NAMES: ReadonlySet<string> = new Set(
  SKILLS_CATALOG.map((entry) => entry.name),
);

/** Catalog skills whose `defaultEnabled` flag is set — the curated default. */
export const RECOMMENDED_SKILL_NAMES: ReadonlySet<string> = new Set(
  DEFAULT_ENABLED_SKILL_NAMES,
);

export function isCatalogSkill(name: string): boolean {
  return CATALOG_SKILL_NAMES.has(name);
}

export function isRecommendedSkill(name: string): boolean {
  return RECOMMENDED_SKILL_NAMES.has(name);
}

/**
 * The two lists that decide which skills a new conversation starts with.
 *
 * They cover disjoint populations on purpose:
 *
 * - `enabledSkills` is an **allow-list over the bundled catalog**. The catalog
 *   is a build-time snapshot of ~60 skills, so a deny-list there means every
 *   future addition is on by default for everyone — the behaviour reported in
 *   OpenHands#16302. `undefined` is the "never migrated" sentinel and must
 *   survive settings hydration; see {@link migrateSkillEnablement}.
 * - `disabledSkills` stays a **deny-list over user- and project-authored
 *   skills**. Those are discovered at runtime from `.agents/skills/`, and a
 *   skill the user wrote themselves should be on the moment it appears.
 */
export interface SkillEnablement {
  enabledSkills?: string[];
  disabledSkills?: string[];
}

/**
 * The catalog allow-list to apply, falling back to the curated default for a
 * workspace that has never persisted one.
 */
export function resolveEnabledCatalogSkills(
  enablement: SkillEnablement,
): string[] {
  return enablement.enabledSkills ?? [...DEFAULT_ENABLED_SKILL_NAMES];
}

export function resolveEnabledCatalogSkillSet(
  enablement: SkillEnablement,
): Set<string> {
  return new Set(resolveEnabledCatalogSkills(enablement));
}

/**
 * Whether a single skill is active for new conversations.
 *
 * The deny-list is still honoured for catalog skills, which only matters
 * before the migration has run: until then a pre-existing "I turned this off"
 * lives in `disabledSkills` alone, and the allow-list fallback would otherwise
 * silently switch it back on.
 */
export function isSkillEnabled(
  name: string,
  enablement: SkillEnablement,
): boolean {
  const disabled = enablement.disabledSkills ?? [];
  if (disabled.includes(name)) return false;
  if (!isCatalogSkill(name)) return true;
  return resolveEnabledCatalogSkillSet(enablement).has(name);
}

/**
 * One-shot conversion of a workspace from "all catalog skills on, minus a
 * deny-list" to an explicit allow-list.
 *
 * Returns `undefined` once the workspace is already migrated.
 *
 * A fresh workspace is migrated too, even though the resolver's fallback
 * already gives it the same set: persisting an *explicit* list is what stops a
 * later catalog addition marked `defaultEnabled` from switching itself on in a
 * workspace that has already been initialised.
 */
export function migrateSkillEnablement(
  enablement: SkillEnablement,
): { enabled_skills: string[]; disabled_skills: string[] } | undefined {
  if (enablement.enabledSkills !== undefined) return undefined;

  const disabled = enablement.disabledSkills ?? [];
  // A deny-list naming a catalog skill is the only positive evidence that this
  // workspace predates the allow-list and had the old "everything on" default.
  // A deny-list holding local skill names alone says nothing about catalog
  // preferences, so it is treated as a fresh workspace.
  const isExistingWorkspace = disabled.some(isCatalogSkill);

  const enabled = isExistingWorkspace
    ? SKILLS_CATALOG.map((entry) => entry.name).filter(
        (name) => !disabled.includes(name),
      )
    : [...DEFAULT_ENABLED_SKILL_NAMES];

  return {
    enabled_skills: enabled,
    // Catalog names have moved to the allow-list; leaving them here too would
    // let a stale deny entry veto a skill the user later switches back on.
    disabled_skills: disabled.filter((name) => !isCatalogSkill(name)),
  };
}

/**
 * Lift the two persisted lists off a settings record.
 *
 * Structurally typed rather than tied to `Settings` so this module stays free
 * of the settings graph and remains a pure, cheaply testable unit.
 */
export function toSkillEnablement(settings: {
  enabled_skills?: string[];
  disabled_skills?: string[];
}): SkillEnablement {
  return {
    enabledSkills: settings.enabled_skills,
    disabledSkills: settings.disabled_skills,
  };
}

/**
 * Slash command → catalog skill.
 *
 * Both forms a user can type are indexed: the commands a skill declares in its
 * own `triggers` (an automation card's launch prompt is one of these — see
 * `findAutomationCommand`), and `/<skill-name>`, which is what the skill
 * detail modal's "Use skill" button inserts.
 */
const CATALOG_SKILL_BY_SLASH_COMMAND: ReadonlyMap<string, string> = new Map(
  SKILLS_CATALOG.flatMap((entry) => {
    const commands = [
      `/${entry.name}`,
      ...(entry.triggers ?? []).filter((trigger) => trigger.startsWith("/")),
    ];
    return commands.map(
      (command) => [command.toLowerCase(), entry.name] as const,
    );
  }),
);

/**
 * The catalog skill a message invokes by name, if any.
 *
 * Typing `/standup-digest:setup` — or clicking the automation card that fills
 * it in — is an explicit request for that skill, so it is loaded for that
 * conversation whatever the stored lists say, and without changing them. The
 * alternative is the worst possible outcome: 18 of the catalog's 24 slash
 * commands are owned by skills that are off by default, so the command would
 * reach the agent as a bare string with none of its instructions, and the card
 * would look like it did nothing.
 *
 * Only the leading token counts. Matching a `/word` anywhere in prose would
 * re-admit most of the catalog through the back door, which is the behaviour
 * the allow-list exists to remove.
 */
export function findInvokedCatalogSkills(query?: string): string[] {
  const firstToken = query?.trim().split(/\s+/, 1)[0];
  if (!firstToken?.startsWith("/")) return [];

  const skill = CATALOG_SKILL_BY_SLASH_COMMAND.get(firstToken.toLowerCase());
  return skill ? [skill] : [];
}
