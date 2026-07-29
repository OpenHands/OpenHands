export function PendingStopIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      className={className}
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        className="fill-[var(--oh-foreground)] transition-colors duration-150 group-hover:fill-[var(--oh-text-secondary)]"
      />
      <rect
        x="9"
        y="9"
        width="6"
        height="6"
        rx="1"
        className="fill-[var(--oh-color-tertiary)]"
      />
    </svg>
  );
}
