import { cn } from "#/utils/utils";

/**
 * Full Tailwind candidates for fine-hover media — devices where CSS `:hover`
 * is a reliable primary interaction (mouse/trackpad). Coarse-pointer /
 * touch-primary devices match the inverse and keep overflow actions always
 * visible + clickable. Kept as complete string literals so the production CSS
 * scanner can emit the arbitrary-variant rules (dynamic prefix concatenation
 * is not discoverable).
 */
export const FINE_HOVER_ACTION_CLASSES = [
  "[@media(hover:hover)_and_(pointer:fine)]:pointer-events-none",
  "[@media(hover:hover)_and_(pointer:fine)]:invisible",
  "[@media(hover:hover)_and_(pointer:fine)]:opacity-0",
  "[@media(hover:hover)_and_(pointer:fine)]:group-hover:pointer-events-auto",
  "[@media(hover:hover)_and_(pointer:fine)]:group-hover:visible",
  "[@media(hover:hover)_and_(pointer:fine)]:group-hover:opacity-100",
  "[@media(hover:hover)_and_(pointer:fine)]:group-focus-within:pointer-events-auto",
  "[@media(hover:hover)_and_(pointer:fine)]:group-focus-within:visible",
  "[@media(hover:hover)_and_(pointer:fine)]:group-focus-within:opacity-100",
] as const;

export const FINE_HOVER_YIELD_CLASSES = [
  "[@media(hover:hover)_and_(pointer:fine)]:opacity-100",
  "[@media(hover:hover)_and_(pointer:fine)]:group-hover:opacity-0",
  "[@media(hover:hover)_and_(pointer:fine)]:group-focus-within:opacity-0",
] as const;

export const FINE_HOVER_RESERVE_CLASSES = [
  "[@media(hover:hover)_and_(pointer:fine)]:min-w-0",
  "[@media(hover:hover)_and_(pointer:fine)]:group-hover:min-w-[3.75rem]",
  "[@media(hover:hover)_and_(pointer:fine)]:group-focus-within:min-w-[3.75rem]",
] as const;

export const FINE_HOVER_PINNED_TIMESTAMP_CLASSES = [
  "[@media(hover:hover)_and_(pointer:fine)]:flex",
  "[@media(hover:hover)_and_(pointer:fine)]:group-hover:hidden",
  "[@media(hover:hover)_and_(pointer:fine)]:group-focus-within:hidden",
] as const;

/**
 * Overlay action chrome (ellipsis, pin, etc.): always interactable on touch;
 * hover/focus-reveal only on fine-pointer hover devices.
 */
export function hoverRevealActionClassName(forceVisible = false): string {
  if (forceVisible) {
    return "pointer-events-auto visible opacity-100";
  }

  return cn(
    "pointer-events-auto visible opacity-100",
    ...FINE_HOVER_ACTION_CLASSES,
  );
}

/**
 * Companion for timestamps that yield space to hover-reveal actions:
 * hidden on touch (actions stay visible); on fine-pointer devices, visible
 * until the row is hovered / focused / menu-open.
 */
export function hoverRevealYieldClassName(forceHidden = false): string {
  if (forceHidden) {
    return "opacity-0";
  }

  return cn("opacity-0", ...FINE_HOVER_YIELD_CLASSES);
}

/**
 * Reserve trailing space for hover-reveal actions. Always reserved on touch;
 * on fine-pointer devices, reserved on hover / focus / open.
 */
export function hoverRevealReserveClassName(forceReserved = false): string {
  if (forceReserved) {
    return "min-w-[3.75rem]";
  }

  return cn("min-w-[3.75rem]", ...FINE_HOVER_RESERVE_CLASSES);
}

/**
 * Absolute-positioned timestamp that sits under a pinned-card ellipsis slot:
 * shown at rest on fine-pointer devices, hidden when the row reveals actions
 * (or when the menu is open / on touch where the ellipsis stays visible).
 */
export function hoverRevealPinnedTimestampClassName(
  forceHidden = false,
): string {
  if (forceHidden) {
    return "hidden";
  }

  return cn("hidden", ...FINE_HOVER_PINNED_TIMESTAMP_CLASSES);
}
