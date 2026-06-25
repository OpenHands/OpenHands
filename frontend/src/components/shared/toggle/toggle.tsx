import { cn } from "#/utils/utils";

interface ToggleProps {
  checked: boolean;
  onClick?: () => void;
  disabled?: boolean;
  "aria-label"?: string;
  title?: string;
}

export function Toggle({
  checked,
  onClick,
  disabled,
  "aria-label": ariaLabel,
  title,
}: ToggleProps) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      className={cn("cursor-pointer", disabled && "cursor-not-allowed")}
      aria-label={ariaLabel}
      disabled={disabled}
      title={title}
    >
      <div
        className={cn(
          "w-12 h-6 rounded-xl flex items-center p-1.5",
          checked && "justify-end bg-white",
          !checked && "justify-start bg-base-secondary",
          disabled && "opacity-50",
        )}
      >
        <div
          className={cn(
            "w-3 h-3 rounded-xl",
            checked ? "bg-[#0D0F11]" : "bg-tertiary-light",
          )}
        />
      </div>
    </button>
  );
}
