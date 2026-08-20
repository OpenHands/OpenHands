import { cn } from "#/utils/utils";
import {
  formControlTransitionClassName,
  formControlMutedHoverClassName,
} from "#/utils/form-control-classes";

/**
 * Shared control for icon-only buttons in mobile top bars (global nav, chat
 * header, full-screen panel chrome). Matches {@link RightPanelToggle}: `p-1`
 * with a `size-5` icon yields a 28×28px hit target.
 */
export const mobileTopBarIconButtonClassName = cn(
  "inline-flex shrink-0 cursor-pointer items-center justify-center rounded-md p-1",
  "text-[var(--oh-muted)]",
  formControlTransitionClassName,
  formControlMutedHoverClassName,
);

/** Lucide / inline SVG wrapper size aligned with `BlockDrawerLeftIcon` (`w-5 h-5`). */
export const mobileTopBarIconClassName = "size-5 shrink-0";
