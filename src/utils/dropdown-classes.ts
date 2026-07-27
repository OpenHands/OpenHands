import { cn } from "#/utils/utils";
import { formControlShellClassName } from "#/utils/form-control-classes";

/** Snap hover colors instantly — no transition-colors delay on menus/dropdowns. */
export const dropdownInstantColorClassName = "transition-none";

/** 2px vertical gap between rows in a dropdown/context menu list. */
export const dropdownMenuListGapClassName = "gap-0.5";

/** Standard horizontal gap between a row icon and its label. */
export const dropdownMenuRowGapClassName = "gap-2";

/** Flex column shell for a dropdown menu item list. */
export const dropdownMenuListClassName = cn(
  "flex flex-col",
  dropdownMenuListGapClassName,
);

/** Combobox/select trigger shell with instant hover colors. */
export const dropdownTriggerShellClassName = cn(
  "bg-tertiary border border-[var(--oh-border-input)] rounded p-2",
  "flex items-center gap-2",
  formControlShellClassName,
  dropdownInstantColorClassName,
  "group w-full gap-2 px-3 text-[var(--oh-muted)] hover:text-white",
);

/** Standard white-label menu row. */
export const dropdownMenuRowClassName = cn(
  "group flex w-full cursor-pointer items-center rounded px-2 py-2 text-left text-sm font-normal text-white",
  dropdownMenuRowGapClassName,
  "hover:bg-[var(--oh-interactive-hover)] disabled:cursor-not-allowed disabled:opacity-60",
  dropdownInstantColorClassName,
);

/** Menu row using foreground token (context menus). */
export const dropdownMenuRowForegroundClassName = cn(
  "group flex w-full cursor-pointer items-center rounded px-2 py-2 text-start text-sm font-normal",
  dropdownMenuRowGapClassName,
  "text-[var(--oh-foreground)] hover:bg-[var(--oh-interactive-hover)]",
  "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent",
  dropdownInstantColorClassName,
);

/** Icon inside a menu row — muted until row hover/focus. */
export const dropdownMenuRowIconClassName = cn(
  "shrink-0 text-[var(--oh-muted)] group-hover:text-white group-focus-visible:text-white",
  dropdownInstantColorClassName,
);

/** Wrapper for SVG/React icon nodes inside a menu row. */
export const dropdownMenuRowIconWrapperClassName = cn(
  "flex size-4 shrink-0 items-center justify-center [&_svg]:text-current",
  dropdownMenuRowIconClassName,
);

/** Enum/filter dropdown trigger chip. */
export const dropdownFilterTriggerClassName = cn(
  "inline-flex cursor-pointer items-center gap-1.5 rounded-lg border px-3 py-2 text-sm font-medium",
  "border-[var(--oh-border)] bg-base-secondary text-white",
  "focus-visible:border-white/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/20",
  dropdownInstantColorClassName,
);

/**
 * Scroll cap for the switch-profile menu: four two-line rows
 * (p-2 + leading-5 title + gap-0.5 + leading-4 subtitle) plus list gaps.
 */
export const switchProfileMenuListScrollClassName = cn(
  dropdownMenuListClassName,
  "overflow-y-auto custom-scrollbar-always",
  "max-h-[13.875rem]",
);

/**
 * Tall sidebar/context menus: cap height to the viewport and scroll with the
 * custom scrollbar when content overflows (no scrollbar when everything fits).
 */
export const dropdownMenuViewportScrollClassName = cn(
  "overflow-y-auto custom-scrollbar",
  "max-h-[min(60vh,calc(100dvh-5rem))]",
);

/** Footer action row inside a dropdown panel. */
export const dropdownFooterActionClassName = cn(
  "group flex w-full items-center rounded-md px-2 py-2 text-sm font-normal text-white",
  dropdownMenuRowGapClassName,
  "hover:bg-[var(--oh-interactive-hover)]",
  dropdownInstantColorClassName,
);
