import type { ReactNode } from "react";
import { cn } from "#/utils/utils";

interface SegmentedToggleOption<T extends string> {
  value: T;
  label: string;
  icon?: ReactNode;
}

interface SegmentedToggleProps<T extends string> {
  value: T;
  options: SegmentedToggleOption<T>[];
  onChange: (value: T) => void;
  ariaLabel: string;
  testId?: string;
  className?: string;
  /** Stretch the control and give each option an equal share of the width. */
  equalWidth?: boolean;
}

/**
 * Lightweight 2-state segmented control used for the files-tab toggles
 * ("Rich"/"Plain") and a few other compact two-way choices.
 */
export function SegmentedToggle<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  testId,
  className,
  equalWidth = false,
}: SegmentedToggleProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      data-testid={testId}
      className={cn(
        equalWidth ? "flex w-full" : "inline-flex",
        "items-center rounded-md bg-[var(--oh-surface-raised)] p-0.5 text-xs",
        className,
      )}
    >
      {options.map((option) => {
        const isActive = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={isActive}
            data-testid={
              testId ? `${testId}-option-${option.value}` : undefined
            }
            onClick={() => onChange(option.value)}
            className={cn(
              "inline-flex items-center gap-1.5 px-2 py-0.5 rounded cursor-pointer transition-colors",
              equalWidth && "flex-1 justify-center text-center",
              isActive
                ? "bg-[var(--oh-interactive-hover)] text-white"
                : "text-[var(--oh-muted)] hover:text-white",
            )}
          >
            {option.icon}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
