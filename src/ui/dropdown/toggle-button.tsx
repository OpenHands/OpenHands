import {
  ComboboxCaretIcon,
  comboboxCaretButtonClassName,
} from "#/ui/combobox-caret";
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
        className: cn(
          comboboxCaretButtonClassName,
          "text-current",
          isOpen && "rotate-180",
          isDisabled && "cursor-not-allowed",
        ),
      })}
    >
      <ComboboxCaretIcon />
    </button>
  );
}
