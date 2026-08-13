import type { ComponentType } from "react";
import type { WebClientFeatureFlags } from "#/api/option-service/option.types";

/**
 * Facts a contributed settings surface (a section or a nav/page entry) may gate
 * its visibility on. Deliberately small and host-owned: these are things the
 * app already derives for its built-ins. Evaluating a `when` predicate never
 * runs contributor code — it only reads these host facts.
 *
 * Kept intentionally close in spirit to the experimental extension host
 * UI-context (`src/extensions/ui-context.tsx` in the closed PR #1659) so a
 * later declarative `when`-clause producer can read the same fact set.
 *
 * The fact set grows here (and only here); every consumer sees a single,
 * consistent source of truth via {@link useSettingsContext}. Role/permission
 * facts are the obvious next addition.
 */
export interface SettingsContext {
  /** Active backend kind. */
  backendKind: "local" | "cloud";
  /** Active Cloud organization id, or `null` for local / no org selected. */
  orgId: string | null;
  /** Web-client feature flags, or `undefined` before config resolves. */
  featureFlags: WebClientFeatureFlags | undefined;
}

/**
 * A contributed settings section.
 *
 * The `{ id, page, order, when }` envelope is the stable contract; `Component`
 * is the *producer* and the only part expected to change across future phases
 * (a native React component today, a REST-served manifest or sandboxed bundle
 * later). Keeping the renderable at the edge lets the host stay unchanged when
 * the producer changes.
 *
 * Sections own their own persistence (section-owned save): a section reads and
 * writes only the settings fields it understands, so the host never needs to
 * know which fields a section touches. This is what lets a section — including,
 * eventually, a plugin-contributed one — save a field the host has never heard
 * of.
 */
export interface SettingsSection {
  /** Stable, namespaced identifier, e.g. `"app.git"`. */
  id: string;
  /** Settings route the section renders on, e.g. `"/settings/app"`. */
  page: string;
  /** Sort order within the page (ascending). */
  order: number;
  /** Optional visibility predicate. Omitted means always visible. */
  when?: (context: SettingsContext) => boolean;
  /** The section body. Rendered as-is by the host. */
  Component: ComponentType;
}

const sections = new Map<string, SettingsSection>();

/**
 * Register a settings section. Idempotent by `id`: re-registering the same id
 * replaces the previous entry rather than duplicating it, which keeps module
 * re-evaluation (HMR, repeated test imports) from stacking duplicates.
 */
export function registerSettingsSection(section: SettingsSection): void {
  sections.set(section.id, section);
}

/**
 * Return the visible sections for a page, sorted by `order` then `id` (stable
 * tiebreak). A section with no `when` is always visible; a `when` that throws
 * is treated as hidden so a misbehaving predicate cannot break the page.
 */
export function getSettingsSections(
  page: string,
  context: SettingsContext,
): SettingsSection[] {
  const isVisible = (section: SettingsSection): boolean => {
    if (!section.when) return true;
    try {
      return section.when(context);
    } catch {
      return false;
    }
  };

  return Array.from(sections.values())
    .filter((section) => section.page === page && isVisible(section))
    .sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
}

/** Remove all registered sections. Intended for tests. */
export function clearSettingsSections(): void {
  sections.clear();
}
