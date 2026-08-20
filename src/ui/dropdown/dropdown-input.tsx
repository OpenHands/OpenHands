import { cn } from "#/utils/utils";
import { formControlInlineInputClassName } from "#/utils/form-control-classes";

interface DropdownInputProps {
  placeholder?: string;
  isDisabled: boolean;
  getInputProps: (props?: object) => object;
  /** When false, placeholder hint keeps upright type (e.g. backend selector). */
  italicPlaceholder?: boolean;
  fitContent?: boolean;
}

export function DropdownInput({
  placeholder,
  isDisabled,
  getInputProps,
  italicPlaceholder = true,
  fitContent = false,
}: DropdownInputProps) {
  return (
    <input
      {...getInputProps({
        placeholder,
        disabled: isDisabled,
        className: cn(
          "outline-none bg-transparent text-white not-italic",
          fitContent
            ? "w-auto field-sizing-content whitespace-nowrap text-sm"
            : "flex-1 min-w-0",
          italicPlaceholder &&
            "placeholder:italic placeholder:text-tertiary-alt",
          !italicPlaceholder && "placeholder:text-tertiary-alt",
          formControlInlineInputClassName,
          "px-0 not-italic text-inherit",
        ),
      })}
    />
  );
}
