import type { Automation } from "#/types/automation";

/** How the list page sections its automations; the first is the default. */
export const AUTOMATION_GROUP_BY_VALUES = ["status", "tag"] as const;

export type AutomationGroupBy = (typeof AUTOMATION_GROUP_BY_VALUES)[number];

export const AUTOMATIONS_GROUP_BY_STORAGE_KEY =
  "openhands-automations-group-by";

export function readStoredAutomationGroupBy(): AutomationGroupBy {
  if (typeof window === "undefined") {
    return "status";
  }

  const stored = window.localStorage.getItem(AUTOMATIONS_GROUP_BY_STORAGE_KEY);
  return stored === "tag" ? "tag" : "status";
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
