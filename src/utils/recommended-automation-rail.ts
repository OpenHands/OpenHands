import {
  AUTOMATION_CATALOG,
  type RecommendedAutomation,
} from "@openhands/extensions/automations";
import { getFeaturedAutomationIds } from "#/manifests/automation-interface";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import type { Automation } from "#/types/automation";

export function getAutomationsByPopularity(
  catalog: RecommendedAutomation[],
): RecommendedAutomation[] {
  return catalog
    .map((automation, index) => ({ automation, index }))
    .sort((a, b) => {
      const byPopularity =
        (b.automation.popularityRank ?? 0) - (a.automation.popularityRank ?? 0);
      return byPopularity || a.index - b.index;
    })
    .map(({ automation }) => automation);
}

/**
 * Slug used to match a catalog entry against an installed automation name.
 * Catalog ids are already kebab-case; installed names are human titles.
 */
export function normalizeAutomationKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function catalogMatchKeys(entry: RecommendedAutomation): string[] {
  return [entry.id, entry.name, entry.skill]
    .filter((value): value is string => Boolean(value))
    .map(normalizeAutomationKey);
}

export function isCatalogAutomationAdded(
  entry: RecommendedAutomation,
  installed: readonly Pick<Automation, "name">[],
): boolean {
  const installedKeys = new Set(
    installed.map((automation) => normalizeAutomationKey(automation.name)),
  );
  return catalogMatchKeys(entry).some((key) => installedKeys.has(key));
}

function isProvenAutomation(entry: RecommendedAutomation): boolean {
  return getFeaturedAutomationIds().includes(entry.id);
}

/** Catalog cards that launch a conversation instead of a host setup form. */
export function isConversationLaunchAutomation(
  entry: RecommendedAutomation,
): boolean {
  return SETUP_REGISTRY.findById(entry.id) == null;
}

export interface RecommendedRailGroups {
  proven: RecommendedAutomation[];
  conversation: RecommendedAutomation[];
}

/**
 * Home / dashboard rail: remaining proven workflows, then other useful
 * automations that open in a new conversation. Already-created automations
 * are dropped from both groups.
 */
export function getRecommendedRailGroups(
  installed: readonly Pick<Automation, "name">[],
): RecommendedRailGroups {
  // Every catalog entry is offerable: one that declares no integration needs
  // nothing connected, and one that names an integration this host cannot
  // resolve stays visible so the drift is actionable rather than hidden.
  const available = getAutomationsByPopularity(AUTOMATION_CATALOG).filter(
    (entry) => !isCatalogAutomationAdded(entry, installed),
  );

  return {
    proven: available.filter(isProvenAutomation),
    conversation: available.filter(
      (entry) =>
        !isProvenAutomation(entry) && isConversationLaunchAutomation(entry),
    ),
  };
}

export function flattenRecommendedRailGroups(
  groups: RecommendedRailGroups,
): RecommendedAutomation[] {
  return [...groups.proven, ...groups.conversation];
}
