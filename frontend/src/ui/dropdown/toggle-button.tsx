/* eslint-disable react/jsx-props-no-spreading */
import { ChevronDown } from "lucide-react";
import { cn } from "#/utils/utils";

interface ToggleButtonProps {
  isOpen: boolean;
  isDisabled: boolean;
  getToggleButtonProps: (props?: object) => object;
}

export function ToggleButton({
  isOpen,
  isDisabled,
  getToggleButtonProps,
}: ToggleButtonProps) {
  return (
    <button
      type="button"
      data-testid="dropdown-trigger"
      {...getToggleButtonProps({
        disabled: isDisabled,
        className: cn("text-white", isDisabled && "cursor-not-allowed"),
      })}
    >
      <ChevronDown
        size={16}
        className={cn("transition-transform", isOpen && "rotate-180")}
      />
    </button>
  );
}
