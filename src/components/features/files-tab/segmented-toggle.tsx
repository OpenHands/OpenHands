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
}

/**
 * Lightweight 2-state segmented control used for the files-tab toggles
 * ("Diff view" on/off, "Rich"/"Plain"). Kept local because the existing
 * shared switch components are heavier than what we need here.
 */
export function SegmentedToggle<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  testId,
}: SegmentedToggleProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      data-testid={testId}
      className="inline-flex items-center rounded-md bg-[var(--oh-surface-raised)] p-0.5 text-xs"
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
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded cursor-pointer transition-colors",
              isActive
                ? "bg-[var(--oh-interactive-hover)] text-white"
                : "text-[var(--oh-muted)] hover:text-white",
            )}
          >
            {option.icon ? (
              <span className="inline-flex size-3.5 shrink-0 items-center justify-center [&_svg]:size-3.5">
                {option.icon}
              </span>
            ) : null}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
