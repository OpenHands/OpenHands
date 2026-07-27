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

export {
  tableContainerClassName as automationListTableClassName,
  tableRowInteractiveClassName as automationListRowClassName,
  tableCellClassName as automationListCellClassName,
} from "#/utils/table-row-classes";
