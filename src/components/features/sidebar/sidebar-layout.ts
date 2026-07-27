import { cn } from "#/utils/utils";

/** Nav rows and side nav links — hover color/background snap instantly (no fade). */
export const navInteractiveTransitionClassName =
  "transition-none motion-reduce:transition-none";

/** Expanded sidebar icon column beside labels (matches 36px nav rows). */
export const SIDEBAR_ICON_SLOT_CLASS =
  "flex h-9 w-[18px] shrink-0 items-center justify-center";

/** Collapsed rail: 36px-tall hit target; width follows the row (full rail). */
export const SIDEBAR_COLLAPSED_ICON_SLOT_CLASS =
  "relative h-9 min-h-9 max-h-9 w-full shrink-0";

export const SIDEBAR_HEADER_ROW_CLASS =
  "flex h-10 min-h-10 shrink-0 items-center gap-2 pl-2.5 pr-2.5 w-full";

export function sidebarHeaderRowClassName(collapsed: boolean): string {
  return cn(
    "flex h-10 min-h-10 shrink-0 items-center w-full",
    collapsed ? "px-0" : "gap-2 pl-2.5 pr-2.5",
  );
}

export const SIDEBAR_ROW_INTERACTIVE_CLASS = {
  active: "bg-tertiary text-white font-normal",
  idle: "text-[var(--oh-muted)] hover:text-white hover:bg-[var(--oh-surface-raised)]",
} as const;

export function sidebarNavListClassName(collapsed: boolean): string {
  return cn(
    "flex flex-col gap-0.5 w-full shrink-0 items-stretch",
    !collapsed && "pr-2.5",
  );
}

export function sidebarNavRowClassName(options?: {
  indent?: boolean;
  collapsed?: boolean;
}): string {
  const { indent = false, collapsed = false } = options ?? {};
  return cn(
    "flex h-9 min-h-9 min-w-0 items-center rounded-md",
    navInteractiveTransitionClassName,
    "text-sm leading-5 w-full",
    collapsed
      ? "group gap-0 px-0 overflow-visible bg-transparent hover:bg-transparent"
      : "gap-2 px-2.5 overflow-hidden",
    indent && !collapsed && "pl-7",
  );
}

export function sidebarCollapsedIconBgClassName(active: boolean): string {
  return cn(
    "pointer-events-none absolute inset-0 z-0 rounded-md",
    navInteractiveTransitionClassName,
    active
      ? "bg-tertiary"
      : "bg-transparent group-hover:bg-[var(--oh-surface-raised)]",
  );
}

/** Matches expanded row `px-2.5` + 18px icon column alignment. */
export function sidebarCollapsedIconGlyphClassName(active: boolean): string {
  return cn(
    // Full width inside the 36px-tall slot; `pl-2.5` aligns with expanded `px-2.5` rows.
    // Do not set a narrow `w-[18px]` here — with horizontal padding it shrinks the glyph.
    "relative z-[1] flex h-full w-full items-center justify-start pl-2.5 [&_svg]:shrink-0",
    active
      ? "text-white font-normal"
      : "text-[var(--oh-muted)] group-hover:text-white",
  );
}

export function sidebarNavLabelClassName(collapsed: boolean): string {
  if (collapsed) {
    return "sr-only";
  }
  return "min-w-0 truncate";
}

export const SIDEBAR_ICON_BUTTON_CLASS = cn(
  "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
  navInteractiveTransitionClassName,
  "cursor-pointer",
);

/** Logo + expand overlay when the desktop rail is collapsed. */
export const SIDEBAR_COLLAPSED_LOGO_WRAPPER_CLASS = cn(
  "relative hidden md:block shrink-0 overflow-visible",
  SIDEBAR_COLLAPSED_ICON_SLOT_CLASS,
);

export const SIDEBAR_COLLAPSE_TOGGLE_OVERLAY_CLASS = cn(
  "absolute left-1/2 top-1/2 hidden h-7 w-7 -translate-x-1/2 -translate-y-1/2 md:inline-flex",
  "items-center justify-center rounded-md",
  navInteractiveTransitionClassName,
  "cursor-pointer",
  "text-[var(--oh-muted)] hover:text-white hover:bg-[var(--oh-surface-raised)]",
);
