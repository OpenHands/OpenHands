import { cn } from "#/utils/utils";
import { dropdownInstantColorClassName } from "#/utils/dropdown-classes";

export const automationIconActionButtonClassName = cn(
  "inline-flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-muted hover:bg-interactive-hover hover:text-white focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-muted",
  dropdownInstantColorClassName,
);

/** Text + icon Run now control on automation grid cards (matches kebab height). */
export const automationRunNowTextButtonClassName = cn(
  "inline-flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-md border-0 bg-transparent px-2 text-xs text-muted hover:bg-interactive-hover hover:text-white focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-muted",
  dropdownInstantColorClassName,
);
