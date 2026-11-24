import type { AutomationKvEntry } from "#/types/automation";

export interface KvChange {
  key: string;
  value: unknown;
}

export interface ResolvedKvDrafts {
  /** Drafts that parse and differ from the stored value. */
  changes: KvChange[];
  /** Drafts that differ in text but are not valid JSON. */
  invalidKeys: ReadonlySet<string>;
}

export function formatKvValue(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

/**
 * Sorts edited key texts into writable changes and invalid ones. A draft that
 * parses to the stored value (a reformat or whitespace edit) is neither.
 */
export function resolveKvDrafts(
  entries: AutomationKvEntry[],
  drafts: Readonly<Record<string, string>>,
): ResolvedKvDrafts {
  const changes: KvChange[] = [];
  const invalidKeys = new Set<string>();

  entries.forEach(({ key, value }) => {
    const draft = drafts[key];
    if (draft === undefined || draft === formatKvValue(value)) return;
    try {
      const parsed: unknown = JSON.parse(draft);
      if (JSON.stringify(parsed) !== JSON.stringify(value)) {
        changes.push({ key, value: parsed });
      }
    } catch {
      invalidKeys.add(key);
    }
  });

  return { changes, invalidKeys };
}
