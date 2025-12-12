import { cn } from "#/utils/utils";
import ChevronDownSmallIcon from "#/icons/chevron-down-small.svg?react";

interface ToggleButtonProps {
  isOpen: boolean;
  disabled: boolean;
  getToggleButtonProps: (
    props?: Record<string, unknown>,
  ) => Record<string, unknown>;
  iconClassName?: string;
}

export function ToggleButton({
  isOpen,
  disabled,
  getToggleButtonProps,
  iconClassName,
}: ToggleButtonProps) {
  return (
    <button
      // eslint-disable-next-line react/jsx-props-no-spreading
      {...getToggleButtonProps({
        disabled,
        className: cn(
          "text-[#fff]",
          "disabled:cursor-not-allowed disabled:opacity-60",
        ),
      })}
      type="button"
      aria-label="Toggle menu"
    >
      <ChevronDownSmallIcon
        className={cn(
          "w-4 h-4 transition-transform",
          isOpen && "rotate-180",
          iconClassName,
        )}
      />
    </button>
  );
}
