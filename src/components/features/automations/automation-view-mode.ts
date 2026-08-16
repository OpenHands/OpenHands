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
  "group relative flex items-stretch transition-colors hover:bg-[var(--oh-interactive-hover)] has-[:focus-visible]:bg-[var(--oh-interactive-hover)]";
