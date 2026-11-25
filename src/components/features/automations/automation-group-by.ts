import type { Automation } from "#/types/automation";

/** How the list page sections its automations; the first is the default. */
export const AUTOMATION_GROUP_BY_VALUES = ["status", "tag", "owner"] as const;

export type AutomationGroupBy = (typeof AUTOMATION_GROUP_BY_VALUES)[number];

export const AUTOMATIONS_GROUP_BY_STORAGE_KEY =
  "openhands-automations-group-by";

export function readStoredAutomationGroupBy(): AutomationGroupBy {
  if (typeof window === "undefined") {
    return "status";
  }

  const stored = window.localStorage.getItem(AUTOMATIONS_GROUP_BY_STORAGE_KEY);
  return (
    AUTOMATION_GROUP_BY_VALUES.find((value) => value === stored) ?? "status"
  );
}

export function writeStoredAutomationGroupBy(groupBy: AutomationGroupBy): void {
  window.localStorage.setItem(AUTOMATIONS_GROUP_BY_STORAGE_KEY, groupBy);
}

export interface AutomationTagGroups {
  /** One group per tag, in alphabetical order. */
  tagged: { tag: string; automations: Automation[] }[];
  untagged: Automation[];
}

/**
 * Sections automations by tag. An automation with several tags is in each of
 * their groups, and every group keeps the order the automations came in.
 */
export function groupAutomationsByTag(
  automations: readonly Automation[],
): AutomationTagGroups {
  const byTag = new Map<string, Automation[]>();
  const untagged: Automation[] = [];

  automations.forEach((automation) => {
    if (!automation.tags?.length) {
      untagged.push(automation);
      return;
    }
    automation.tags.forEach((tag) => {
      const group = byTag.get(tag);
      if (group) {
        group.push(automation);
      } else {
        byTag.set(tag, [automation]);
      }
    });
  });

  return {
    tagged: Array.from(byTag, ([tag, tagAutomations]) => ({
      tag,
      automations: tagAutomations,
    })).sort((a, b) => a.tag.localeCompare(b.tag)),
    untagged,
  };
}

export interface AutomationOwnerGroup {
  /** The person or service account the automations run as; null when unknown. */
  ownerId: string | null;
  isServiceAccount: boolean;
  automations: Automation[];
}

/**
 * Sections automations by the identity they run as. Groups, and the automations
 * in each, are in the order they first came in.
 */
export function groupAutomationsByOwner(
  automations: readonly Automation[],
): AutomationOwnerGroup[] {
  const byOwner = new Map<string | null, AutomationOwnerGroup>();

  automations.forEach((automation) => {
    const ownerId = automation.user_id ?? null;
    const group = byOwner.get(ownerId);
    if (group) {
      group.automations.push(automation);
    } else {
      byOwner.set(ownerId, {
        ownerId,
        isServiceAccount: automation.run_as_kind === "service",
        automations: [automation],
      });
    }
  });

  return Array.from(byOwner.values());
}
