/** Bordered table container on base-secondary surfaces. */
export const tableContainerClassName =
  "overflow-hidden rounded-md border border-[var(--oh-border)] bg-base-secondary";

/** Fixed 44px row height for compact data tables (secrets, automations, MCP). */
export const tableRowHeightClassName = "h-11";

export const tableRowClassName = [
  tableRowHeightClassName,
  "border-t border-[var(--oh-border)] transition-colors",
].join(" ");

/** Subtle row highlight on base-secondary tables. */
export const tableRowHoverClassName =
  "hover:bg-interactive-hover-low outline-none focus:outline-none focus-visible:outline-none focus-visible:bg-interactive-hover-low";

export const tableCellClassName = "px-3 align-middle";

export const tableRowInteractiveClassName = [
  tableRowClassName,
  tableRowHoverClassName,
  "cursor-pointer",
].join(" ");
