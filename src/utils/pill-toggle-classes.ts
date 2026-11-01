import { cn } from "#/utils/utils";

/**
 * Borderless pill that toggles on and off, shared by the Templates filter
 * chips and the setup choice tabs. Callers add their own layout classes.
 */
export function pillToggleClassName(active: boolean) {
  return cn(
    "inline-flex cursor-pointer items-center gap-2 whitespace-nowrap rounded-full border-0 px-3 py-1.5 text-sm transition-colors",
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus",
    active
      ? "bg-contrast text-contrast-foreground"
      : "bg-contrast/5 text-tertiary-light hover:bg-surface-raised hover:text-contrast",
  );
}
