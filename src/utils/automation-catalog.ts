import {
  AUTOMATION_CATALOG,
  type RecommendedAutomation,
} from "@openhands/extensions/automations";
import {
  SKILLS_CATALOG,
  type SkillCatalogEntry,
} from "@openhands/extensions/skills";
import type { LucideIcon } from "lucide-react";
import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import { interpolateValues } from "#/manifests/manifest-template";
import type { Automation } from "#/types/automation";

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

/**
 * The glyph a card shows in place of its integration logos, or null when it
 * has none to show and the logos speak for it.
 *
 * An entry naming the service it talks to is recognised by that service's
 * logo; one whose identity is the work itself declares a slug instead. The
 * lookup is a guard as much as a mapping — the catalog is published elsewhere,
 * so a slug this host has no artwork for falls back rather than rendering
 * nothing.
 */
export function getAutomationIcon(
  automation: RecommendedAutomation,
): LucideIcon | null {
  const slug = automation.icon;
  if (!slug) return null;
  return (
    (MANIFEST_ICON_BY_SLUG as Record<string, LucideIcon | undefined>)[slug] ??
    null
  );
}

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

const CATALOG_ENTRY_BY_ID = new Map<string, RecommendedAutomation>(
  AUTOMATION_CATALOG.map((entry) => [entry.id, entry]),
);

/** Rejects markup the same way the catalog schema's copy rule does. */
const MARKUP_PATTERN = /<[A-Za-z/!]/;

/** Every `{{` an impact phrase opens must be the count placeholder. */
const FOREIGN_PLACEHOLDER_PATTERN = /\{\{(?!count\}\})/;

const COUNT_PLACEHOLDER = "{{count}}";

interface AutomationImpactCopy {
  one: string;
  other: string;
}

function isImpactPhrase(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    !MARKUP_PATTERN.test(value) &&
    !FOREIGN_PLACEHOLDER_PATTERN.test(value)
  );
}

/**
 * The entry's `impact` value statement, or null when it declares none this
 * host can honor. The catalog is published elsewhere, so the field is read
 * off the raw entry and validated here rather than trusted from the package
 * types — which also keeps this module compiling against a pinned package
 * that predates the field. A basis other than `completed-runs` is a counter
 * this host does not know how to compute, so it renders nothing rather than
 * a number the phrase was not written for.
 */
function readImpactCopy(
  entry: RecommendedAutomation,
): AutomationImpactCopy | null {
  const { impact } = entry as unknown as Record<string, unknown>;
  if (typeof impact !== "object" || impact === null) return null;
  const { basis, one, other } = impact as Record<string, unknown>;
  if (basis !== "completed-runs") return null;
  if (!isImpactPhrase(one) || !isImpactPhrase(other)) return null;
  if (!other.includes(COUNT_PLACEHOLDER)) return null;
  return { one, other };
}

/**
 * The catalog entry an installed automation was created from, via the
 * `template` provenance its setup stored — the only join back to the catalog.
 * Null for automations without provenance: imports, agent-built ones, and
 * entries whose setup publishes no version.
 */
function getAutomationTemplateEntry(
  automation: Automation,
): RecommendedAutomation | null {
  const metadata = automation.preset_metadata;
  if (typeof metadata !== "object" || metadata === null) return null;
  const { template } = metadata as Record<string, unknown>;
  if (typeof template !== "object" || template === null) return null;
  const { id } = template as Record<string, unknown>;
  if (typeof id !== "string" || id.length === 0) return null;
  return CATALOG_ENTRY_BY_ID.get(id) ?? null;
}

/**
 * The value statement an automation card shows for its completed runs, or
 * null when there is nothing defensible to say: no template provenance, no
 * honored `impact` declaration, an unknown count (older service with more
 * history than the sample), or a count of zero — absence, never a zero-value
 * claim. Provenance persists across prompt edits, so a reworked automation
 * keeps its template's phrase; the phrases are run-shaped, which keeps them
 * true for as long as the automation runs at all.
 */
export function resolveAutomationImpactStatement(
  automation: Automation,
  completedTotal: number | null,
): string | null {
  if (completedTotal === null || completedTotal < 1) return null;
  const entry = getAutomationTemplateEntry(automation);
  if (!entry) return null;
  const copy = readImpactCopy(entry);
  if (!copy) return null;
  const phrase = completedTotal === 1 ? copy.one : copy.other;
  return interpolateValues(phrase, { count: completedTotal.toLocaleString() });
}
