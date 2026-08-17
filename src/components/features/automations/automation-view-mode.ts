export type AutomationViewMode = "grid" | "list";

export const AUTOMATIONS_VIEW_MODE_STORAGE_KEY = "openhands-automations-view";

export function readStoredAutomationViewMode(): AutomationViewMode {
  if (typeof window === "undefined") {
    return "grid";
  }

  const stored = window.localStorage.getItem(AUTOMATIONS_VIEW_MODE_STORAGE_KEY);
  return stored === "list" ? "list" : "grid";
}

export function writeStoredAutomationViewMode(view: AutomationViewMode): void {
  window.localStorage.setItem(AUTOMATIONS_VIEW_MODE_STORAGE_KEY, view);
}

/** Shared chrome for the dashboard list and the home activity list. */
export const automationActivityListClassName =
  "divide-y divide-[var(--oh-border-subtle)] overflow-hidden rounded-xl border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)]";

export const automationActivityRowClassName =
  "group relative flex items-stretch transition-colors hover:bg-surface-raised has-[:focus-visible]:bg-surface-raised";

/** Inset last-run strip used under the trigger/sparkline row. */
export const automationCardStatusStripClassName =
  "mt-3 flex min-h-9 items-center justify-between gap-2 overflow-hidden rounded-md border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)] px-3 py-2 text-xs";
