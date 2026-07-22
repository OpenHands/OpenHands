import { cn } from "#/utils/utils";

/**
 * Devices where CSS `:hover` is a reliable primary interaction (mouse/trackpad).
 * Coarse-pointer / touch-primary devices match the inverse and should keep
 * overflow actions always visible + clickable.
 */
export const FINE_HOVER_MEDIA = "(hover: hover) and (pointer: fine)";

/** Tailwind arbitrary-variant prefix scoped to {@link FINE_HOVER_MEDIA}. */
export const fineHoverVariant =
  "[@media(hover:hover)_and_(pointer:fine)]:" as const;

/**
 * True when the primary pointer supports reliable hover (mouse/trackpad).
 * Touch-primary phones/tablets return false so callers can prefer click.
 */
export function canHoverReliably(): boolean {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return false;
  }
  return window.matchMedia(FINE_HOVER_MEDIA).matches;
}

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
    `${fineHoverVariant}pointer-events-none`,
    `${fineHoverVariant}invisible`,
    `${fineHoverVariant}opacity-0`,
    `${fineHoverVariant}group-hover:pointer-events-auto`,
    `${fineHoverVariant}group-hover:visible`,
    `${fineHoverVariant}group-hover:opacity-100`,
    `${fineHoverVariant}group-focus-within:pointer-events-auto`,
    `${fineHoverVariant}group-focus-within:visible`,
    `${fineHoverVariant}group-focus-within:opacity-100`,
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

  return cn(
    "opacity-0",
    `${fineHoverVariant}opacity-100`,
    `${fineHoverVariant}group-hover:opacity-0`,
    `${fineHoverVariant}group-focus-within:opacity-0`,
  );
}

/**
 * Reserve trailing space for hover-reveal actions. Always reserved on touch;
 * on fine-pointer devices, reserved on hover / focus / open.
 */
export function hoverRevealReserveClassName(forceReserved = false): string {
  if (forceReserved) {
    return "min-w-[3.75rem]";
  }

  return cn(
    "min-w-[3.75rem]",
    `${fineHoverVariant}min-w-0`,
    `${fineHoverVariant}group-hover:min-w-[3.75rem]`,
    `${fineHoverVariant}group-focus-within:min-w-[3.75rem]`,
  );
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

  return cn(
    "hidden",
    `${fineHoverVariant}flex`,
    `${fineHoverVariant}group-hover:hidden`,
    `${fineHoverVariant}group-focus-within:hidden`,
  );
}
