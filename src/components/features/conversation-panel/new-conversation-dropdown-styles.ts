import { cn } from "#/utils/utils";
import { dropdownMenuListGapClassName } from "#/utils/dropdown-classes";

/** Match conversation panel filter and other sidebar menus */
export const NEW_CONVERSATION_DROPDOWN_SURFACE = cn(
  "z-50 flex flex-col rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 text-white shadow-lg",
  dropdownMenuListGapClassName,
);
