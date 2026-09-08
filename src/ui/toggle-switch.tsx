import { cn } from "#/utils/utils";

export type ToggleSwitchSize = "md" | "sm";

interface ToggleSwitchVisualProps {
  enabled: boolean;
  /** `sm` is the compact menu-row pill; `md` is the settings/automation switch. */
  size?: ToggleSwitchSize;
  className?: string;
}

/** Shared toggle track + thumb used by settings labels and automation controls. */
export function ToggleSwitchVisual({
  enabled,
  size = "md",
  className,
}: ToggleSwitchVisualProps) {
  const compact = size === "sm";
  return (
    <span
      aria-hidden="true"
      className={cn(
        "relative inline-flex shrink-0 items-center rounded-full",
        "transition-colors duration-200 ease-in-out motion-reduce:transition-none",
        compact ? "h-3.5 w-6 p-[3px]" : "h-[22px] w-[40px] border",
        enabled
          ? compact
            ? "bg-white"
            : "border-white bg-white"
          : compact
            ? "bg-[var(--oh-border)]"
            : "border-[var(--oh-border)] bg-surface-raised",
        className,
      )}
    >
      <span
        className={cn(
          "inline-block rounded-full",
          "transition-transform duration-200 ease-in-out motion-reduce:transition-none",
          compact ? "size-2" : "size-4",
          enabled
            ? compact
              ? "translate-x-[10px] bg-base-secondary"
              : "translate-x-[21px] bg-base-secondary"
            : compact
              ? "translate-x-0 bg-[var(--oh-muted)]"
              : "translate-x-[2px] bg-[var(--oh-muted)]",
        )}
      />
    </span>
  );
}

interface ToggleSwitchProps {
  enabled: boolean;
  label: string;
  onToggle: () => void;
  className?: string;
}

export function ToggleSwitch({
  enabled,
  label,
  onToggle,
  className,
}: ToggleSwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label={label}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
      className={cn("cursor-pointer", className)}
    >
      <ToggleSwitchVisual enabled={enabled} />
    </button>
  );
}
