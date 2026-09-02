import React from "react";
import { Check } from "lucide-react";
import { cn } from "#/utils/utils";
import {
  dropdownMenuRowClassName,
  dropdownMenuRowIconClassName,
} from "#/utils/dropdown-classes";

export function MenuRow({
  icon: Icon,
  label,
  selected,
  variant = "radio",
  onClick,
  testId,
  disabled,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  label: string;
  selected?: boolean;
  /**
   * `"radio"` (default) is a row that's part of a mutually exclusive group
   * (picking one clears the others — e.g. Sort by). `"checkbox"` is an
   * independently toggleable row (e.g. the #15607 grouping toggles) that
   * doesn't imply anything about sibling rows.
   */
  variant?: "radio" | "checkbox";
  onClick: () => void;
  testId?: string;
  disabled?: boolean;
}) {
  // Rows that show a selection checkmark are toggleable preferences, so they
  // get `role="menuitemradio"` when they're part of a mutually exclusive
  // group and `role="menuitemcheckbox"` when each row toggles independently.
  // A row with no `selected` state at all falls back to plain `menuitem`.
  const role =
    selected === undefined
      ? "menuitem"
      : variant === "checkbox"
        ? "menuitemcheckbox"
        : "menuitemradio";
  return (
    <button
      type="button"
      role={role}
      aria-checked={selected === undefined ? undefined : Boolean(selected)}
      data-testid={testId}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "group",
        dropdownMenuRowClassName,
        "text-[var(--oh-foreground)] disabled:opacity-50",
      )}
    >
      <Icon
        className={cn("h-3.5 w-3.5", dropdownMenuRowIconClassName)}
        aria-hidden
      />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {selected ? (
        <Check
          className="ml-auto h-3.5 w-3.5 shrink-0 text-white"
          aria-hidden
        />
      ) : null}
    </button>
  );
}
