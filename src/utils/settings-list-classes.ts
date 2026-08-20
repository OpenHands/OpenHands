import { cn } from "#/utils/utils";
import {
  formControlBorderClassName,
  formControlRadiusClassName,
  formControlSurfaceClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";

/** 48px row height — taller than form controls for settings list/table rows. */
export const settingsListRowHeightClassName = "h-12 min-h-12";

/** Bordered list shell shared by secrets, LLM profiles, and similar settings tables. */
export const settingsListContainerClassName = cn(
  formControlBorderClassName,
  formControlRadiusClassName,
  formControlSurfaceClassName,
  "overflow-hidden",
);

/** Scrollable variant for long settings lists (e.g. secrets). */
export const settingsListScrollContainerClassName = cn(
  settingsListContainerClassName,
  "overflow-auto max-h-[min(70vh,39rem)]",
);

export const settingsListDividerClassName =
  "divide-y divide-[var(--oh-border)]";

/** Interactive row hover on dark list surfaces (base-secondary). */
export const settingsListRowHoverClassName =
  "hover:bg-[var(--oh-interactive-hover-low)]";

export const settingsListRowClassName = cn(
  settingsListRowHeightClassName,
  "flex items-center px-3",
  formControlTransitionClassName,
  settingsListRowHoverClassName,
);

export const settingsListTableRowClassName = cn(
  settingsListRowHoverClassName,
  formControlTransitionClassName,
  "border-t border-[var(--oh-border)] first:border-t-0",
);

export const settingsListTableHeadClassName = cn(
  formControlSurfaceClassName,
  "sticky top-0 z-10 border-b border-[var(--oh-border)]",
);

export const settingsListTableHeaderCellClassName = cn(
  settingsListRowHeightClassName,
  "px-3 text-left text-sm font-medium align-middle",
);

export const settingsListTableCellClassName = cn(
  settingsListRowHeightClassName,
  "px-3 text-sm align-middle min-w-0",
);

export const settingsListIconActionButtonClassName = cn(
  "inline-flex cursor-pointer items-center justify-center rounded-md p-1 text-muted",
  formControlTransitionClassName,
  "hover:bg-[var(--oh-interactive-hover-low)] hover:text-white",
);
