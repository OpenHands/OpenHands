import { cn } from "#/utils/utils";

const CONTEXT_WINDOW_RING_SIZE = 16;
const CONTEXT_WINDOW_RING_STROKE = 2;
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
        stroke="var(--oh-border)"
        strokeWidth={CONTEXT_WINDOW_RING_STROKE}
      />
      <circle
        cx={CONTEXT_WINDOW_RING_SIZE / 2}
        cy={CONTEXT_WINDOW_RING_SIZE / 2}
        r={radius}
        fill="none"
        stroke="white"
        strokeWidth={CONTEXT_WINDOW_RING_STROKE}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={dashOffset}
        transform={`rotate(-90 ${CONTEXT_WINDOW_RING_SIZE / 2} ${CONTEXT_WINDOW_RING_SIZE / 2})`}
        className="transition-[stroke-dashoffset] duration-300"
      />
    </svg>
  );
}
