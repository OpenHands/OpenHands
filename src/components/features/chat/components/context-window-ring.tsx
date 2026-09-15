import { cn } from "#/utils/utils";
import { getContextFillTone } from "#/components/features/conversation/usage-panel/context-meter";

const CONTEXT_WINDOW_RING_SIZE = 16;
const CONTEXT_WINDOW_RING_STROKE = 2;

/**
 * Default opacity of the ring's unfilled track on dark themes.
 *
 * The track is derived from the foreground rather than pinned to a scale stop.
 * It carries information (the arc's proportion is only readable against it), so
 * it is a foreground element, and every stop in the surface family sits close
 * to the surfaces it delimits. Drawing it with `--oh-border` put it in that
 * family: it was 1.57:1 against the composer at rest, and the trigger's hover
 * fill resolves to the same stop, taking it to 1.00:1. No stop in that family
 * fixes it, and no fixed stop holds across the three palettes in
 * `color-themes.ts`, whose scales differ.
 *
 * Light palettes override the foreground and weight tokens because a single
 * translucent mix cannot preserve 3:1 contrast on both dark and light
 * surfaces. `context-window-ring.test.tsx` asserts the effective values for
 * every shipped theme.
 */
export const CONTEXT_WINDOW_RING_TRACK_ALPHA = 0.42;

/** Shared by the ring's track and the popover's usage bar, which had the same defect. */
export const CONTEXT_WINDOW_TRACK_COLOR =
  "color-mix(in srgb, var(--oh-context-window-foreground) var(--oh-context-window-track-weight), transparent)";

const TONE_STROKE = {
  neutral: "var(--oh-context-window-foreground)",
  warning: "#f59e0b", // amber-500
  danger: "#ef4444", // red-500
} as const;

interface ContextWindowRingProps {
  percentage: number;
  className?: string;
}

export function ContextWindowRing({
  percentage,
  className,
}: ContextWindowRingProps) {
  const radius = (CONTEXT_WINDOW_RING_SIZE - CONTEXT_WINDOW_RING_STROKE) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedPercentage = Math.min(100, Math.max(0, percentage));
  const dashOffset = circumference - (clampedPercentage / 100) * circumference;
  const tone = getContextFillTone(clampedPercentage);

  return (
    <svg
      width={CONTEXT_WINDOW_RING_SIZE}
      height={CONTEXT_WINDOW_RING_SIZE}
      viewBox={`0 0 ${CONTEXT_WINDOW_RING_SIZE} ${CONTEXT_WINDOW_RING_SIZE}`}
      className={cn("shrink-0", className)}
      aria-hidden
    >
      <circle
        cx={CONTEXT_WINDOW_RING_SIZE / 2}
        cy={CONTEXT_WINDOW_RING_SIZE / 2}
        r={radius}
        fill="none"
        style={{ stroke: CONTEXT_WINDOW_TRACK_COLOR }}
        strokeWidth={CONTEXT_WINDOW_RING_STROKE}
        data-testid="context-window-ring-track"
      />
      <circle
        cx={CONTEXT_WINDOW_RING_SIZE / 2}
        cy={CONTEXT_WINDOW_RING_SIZE / 2}
        r={radius}
        fill="none"
        stroke={TONE_STROKE[tone]}
        strokeWidth={CONTEXT_WINDOW_RING_STROKE}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={dashOffset}
        transform={`rotate(-90 ${CONTEXT_WINDOW_RING_SIZE / 2} ${CONTEXT_WINDOW_RING_SIZE / 2})`}
        className="transition-[stroke-dashoffset,stroke] duration-300"
        data-testid="context-window-ring-arc"
      />
    </svg>
  );
}
