import { cn } from "#/utils/utils";

/** Matches conversation tabs chrome height; right inset aligns with drawer body `px-4`. */
export const CONVERSATION_SECONDARY_DRAWER_HEADER_CLASSNAME = cn(
  "flex h-10 min-h-10 shrink-0 items-center gap-1.5",
  "border-b border-[var(--oh-border)] py-1 pl-1 pr-4",
);

export const CONVERSATION_SECONDARY_DRAWER_HEADER_ACTION_CLASSNAME = cn(
  "h-7 min-h-7 shrink-0 whitespace-nowrap px-2.5 text-xs",
);

export const CONVERSATION_SECONDARY_DRAWER_CLOSE_BUTTON_CLASSNAME = cn(
  "inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md",
  "text-muted transition-colors hover:bg-white/10 hover:text-white",
);
