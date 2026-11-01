import { cn } from "#/utils/utils";

/** Primary modal heading on dark overlay surfaces. */
export const modalTitleClassName = cn(
  "text-xl leading-6 -tracking-[0.01em] font-medium text-modal-title",
);

/** Compact modal header bar title (folder browser, manage workspaces). */
export const modalTitleSmClassName = cn("text-sm font-medium text-modal-title");

/** Large modal title used by form-style dialogs. */
export const modalTitleLgClassName = cn("text-lg font-medium text-modal-title");

/** Alias of {@link modalTitleLgClassName} for historical call sites. */
export const modalTitleLgMediumClassName = modalTitleLgClassName;
