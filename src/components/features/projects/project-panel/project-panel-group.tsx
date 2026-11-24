import React from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

/**
 * A heading with a count that folds the rows under it, as in "Needs you · 2".
 * Rendered only when it holds something.
 */
export function ProjectPanelGroup({
  id,
  label,
  count,
  defaultOpen = true,
  revealKey = null,
  emphasized = false,
  children,
}: {
  id: string;
  label: string;
  count: number;
  defaultOpen?: boolean;
  /**
   * Set while the group holds something the user is looking at, such as the
   * open conversation. Each time it changes to a new value the group opens,
   * so arriving at a row in a folded group shows it; folding by hand sticks
   * while the value stays the same.
   */
  revealKey?: string | null;
  /** Draw the heading in the foreground color, for groups that need the user. */
  emphasized?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(defaultOpen || revealKey !== null);
  const [revealed, setRevealed] = React.useState(revealKey);
  if (revealed !== revealKey) {
    // Adjust state while rendering, so the row shows without a folded frame.
    setRevealed(revealKey);
    if (revealKey !== null) setOpen(true);
  }
  if (!count) return null;
  const Chevron = open ? ChevronDown : ChevronRight;
  return (
    <section data-testid={`project-panel-group-${id}`} aria-label={label}>
      <button
        type="button"
        aria-expanded={open}
        data-testid={`project-panel-group-${id}-toggle`}
        className={`mb-1 inline-flex items-center gap-1 text-xs ${
          emphasized ? "font-medium text-foreground" : "text-muted"
        } hover:text-foreground`}
        onClick={() => setOpen((value) => !value)}
      >
        {label} · {count}
        <Chevron width={12} height={12} aria-hidden />
      </button>
      {open ? <ul className="flex flex-col gap-0.5">{children}</ul> : null}
    </section>
  );
}
