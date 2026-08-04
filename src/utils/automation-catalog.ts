import type { RecommendedAutomation } from "@openhands/extensions/automations";
import {
  SKILLS_CATALOG,
  type SkillCatalogEntry,
} from "@openhands/extensions/skills";

/**
 * Reading a published automation catalog entry.
 *
 * The catalog states each thing once and leaves the rest to be derived, so an
 * entry no longer carries its own launch command or a flat list of required
 * integration ids. This module is the single place that performs those
 * derivations, so the components consuming the catalog stay unaware of how it
 * is shaped.
 */

const SKILL_BY_NAME = new Map<string, SkillCatalogEntry>(
  SKILLS_CATALOG.map((skill) => [skill.name, skill]),
);

/** Every integration the entry declares, in declaration order. */
export function getIntegrationIds(automation: RecommendedAutomation): string[] {
  return Object.keys(automation.requires.integrations);
}

/**
 * The integrations that gate this automation. One marked `required: false` can
 * be connected later, during setup, so it is still worth showing on the card
 * but must never stand between the user and a launch.
 */
export function getRequiredIntegrationIds(
  automation: RecommendedAutomation,
): string[] {
  return Object.entries(automation.requires.integrations)
    .filter(([, requirement]) => requirement.required !== false)
    .map(([id]) => id);
}

/**
 * The command that invokes this automation's skill, or null when that skill is
 * invoked by description instead.
 *
 * The command is declared once, in the owning skill's own `triggers`; the
 * catalog names that skill only where it differs from the entry id.
 */
export function findAutomationCommand(
  automation: Pick<RecommendedAutomation, "id" | "skill">,
): string | null {
  const skill = SKILL_BY_NAME.get(automation.skill ?? automation.id);
  return skill?.triggers[0] ?? null;
}

/**
 * What launching an automation card hands to the agent.
 *
 * The resolved command is passed through as-is: API routing (host, auth) is
 * discovered by the agent at runtime from `<RUNTIME_SERVICES>` in the system
 * prompt, and the skills themselves carry the instructions for reading that
 * block. An automation whose skill declares no command has the request spelled
 * out instead. Both are instructions to the agent rather than user-facing UI
 * copy, so they are intentionally in English and not localized.
 */
export function getAutomationLaunchPrompt(
  automation: RecommendedAutomation,
): string {
  return (
    findAutomationCommand(automation) ??
    `Set up the ${automation.name} automation`
  );
}
