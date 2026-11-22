import type { MarsSessionPhase } from "#/api/mars/mars-tunnel-backend";
import { cn } from "#/utils/utils";
import { PHASE_DOT_CLASS } from "./managed-agents-labels";

interface SessionStatusDotProps {
  phase: MarsSessionPhase;
  /** Size and placement; defaults to `size-2`. */
  className?: string;
}

/** The colored dot that shows a MARS session's phase at a glance. */
export function SessionStatusDot({ phase, className }: SessionStatusDotProps) {
  return (
    <span
      aria-hidden
      className={cn(
        "size-2 shrink-0 rounded-full",
        PHASE_DOT_CLASS[phase],
        className,
      )}
    />
  );
}
