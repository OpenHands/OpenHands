import type { ReactElement } from "react";
import type { SettingsContext } from "./registry";

/**
 * A contributed settings nav/page entry: one row in the settings navigation and
 * the page it links to. This is the sibling of the settings-section registry
 * ({@link ./registry}) — the section registry owns what renders *inside* a
 * page, this registry owns *which pages exist* and their visibility.
 *
 * The `{ id, to, order, when }` envelope is the stable contract, mirroring the
 * section registry's `{ id, page, order, when }`. `icon`/`text`/`subtitle` are
 * the presentational payload the host renders. Backend-specific (and, later,
 * plugin-contributed) pages register with a `when` predicate instead of the
 * host filtering a hard-coded list with inline `backend.kind`/feature-flag
 * conditionals.
 */
export interface SettingsNavEntry {
  /** Stable, namespaced identifier, e.g. `"page.llm"`. */
  id: string;
  /** Route the entry links to, e.g. `"/settings/llm"`. */
  to: string;
  /** Sort order in the navigation (ascending). */
  order: number;
  /** Nav icon. */
  icon: ReactElement;
  /** i18n key for the nav label / page title. */
  text: string;
  /** i18n key for the short grey subline under the page title. */
  subtitle: string;
  /** Optional visibility predicate. Omitted means always visible. */
  when?: (context: SettingsContext) => boolean;
}

const entries = new Map<string, SettingsNavEntry>();

/**
 * Register a settings nav/page entry. Idempotent by `id`: re-registering the
 * same id replaces the previous entry rather than duplicating it, which keeps
 * module re-evaluation (HMR, repeated test imports) from stacking duplicates.
 */
export function registerSettingsNavEntry(entry: SettingsNavEntry): void {
  entries.set(entry.id, entry);
}

/**
 * Return the visible nav/page entries, sorted by `order` then `id` (stable
 * tiebreak). An entry with no `when` is always visible; a `when` that throws is
 * treated as hidden so a misbehaving predicate cannot break the navigation.
 */
export function getSettingsNavEntries(
  context: SettingsContext,
): SettingsNavEntry[] {
  const isVisible = (entry: SettingsNavEntry): boolean => {
    if (!entry.when) return true;
    try {
      return entry.when(context);
    } catch {
      return false;
    }
  };

  return Array.from(entries.values())
    .filter(isVisible)
    .sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
}

/**
 * Return every registered page route regardless of visibility. Used to tell a
 * "this page exists but is currently hidden" state apart from an unknown path.
 */
export function getRegisteredSettingsNavPaths(): string[] {
  return Array.from(entries.values()).map((entry) => entry.to);
}

/** Remove all registered nav/page entries. Intended for tests. */
export function clearSettingsNavEntries(): void {
  entries.clear();
}
