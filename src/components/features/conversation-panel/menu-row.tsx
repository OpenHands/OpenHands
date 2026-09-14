import React from "react";
import { Check } from "lucide-react";
import { cn } from "#/utils/utils";
import {
  dropdownMenuRowClassName,
  dropdownMenuRowIconClassName,
} from "#/utils/dropdown-classes";
import { ToggleSwitchVisual } from "#/ui/toggle-switch";

export function MenuRow({
  icon: Icon,
  label,
  sublabel,
  selected,
  onClick,
  testId,
  disabled,
  destructive,
  variant = "radio",
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  label: string;
  /** Muted second line under the label (e.g. a threshold hint). */
  sublabel?: string;
  selected?: boolean;
  onClick: () => void;
  testId?: string;
  disabled?: boolean;
  /** Destructive action rows (delete/reset) render in the danger color. */
  destructive?: boolean;
  /**
   * "radio" rows mark selection with a checkmark (mutually exclusive
   * groups); "toggle" rows render a switch pill for independent on/off
   * preferences that stay put after clicking (modal-style menus).
   */
  variant?: "radio" | "toggle";
}) {
  const role =
    selected === undefined
      ? "menuitem"
      : variant === "toggle"
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
        "disabled:opacity-50",
        destructive ? "text-danger" : "text-[var(--oh-foreground)]",
      )}
    >
      <Icon
        className={cn(
          "h-3.5 w-3.5",
          dropdownMenuRowIconClassName,
          destructive &&
            "text-danger group-hover:text-danger group-focus-visible:text-danger",
        )}
        aria-hidden
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate">{label}</span>
        {sublabel ? (
          <span className="block truncate text-[10px] text-[var(--oh-muted)]/70">
            {sublabel}
          </span>
        ) : null}
      </span>
      {selected === undefined ? null : variant === "toggle" ? (
        <ToggleSwitchVisual
          enabled={Boolean(selected)}
          size="sm"
          className="ml-auto"
        />
      ) : selected ? (
        <Check
          className="ml-auto h-3.5 w-3.5 shrink-0 text-white"
          aria-hidden
        />
      ) : null}
    </button>
  );
}
