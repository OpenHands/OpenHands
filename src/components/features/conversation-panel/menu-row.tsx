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
  onClick,
  testId,
  disabled,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  label: string;
  selected?: boolean;
  onClick: () => void;
  testId?: string;
  disabled?: boolean;
}) {
  // Rows that show a selection checkmark are toggleable preferences, so
  // they get `role="menuitemradio"` when they're part of a selectable
  // group and `role="menuitemcheckbox"` when they're a standalone toggle.
  // For simplicity we use `menuitemradio` whenever `selected` is provided
  // (every selectable row in this menu is part of a mutually exclusive
  // group in practice) and fall back to plain `menuitem` otherwise.
  const role = selected === undefined ? "menuitem" : "menuitemradio";
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
