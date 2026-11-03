import { cn } from "#/utils/utils";
import { formControlInlineInputClassName } from "#/utils/form-control-classes";

interface DropdownInputProps {
  placeholder?: string;
  /** Full selected label, shown on hover when the input truncates it. */
  title?: string;
  isDisabled: boolean;
  getInputProps: (props?: object) => object;
  /** When false, placeholder hint keeps upright type (e.g. backend selector). */
  italicPlaceholder?: boolean;
  fitContent?: boolean;
}

export function DropdownInput({
  placeholder,
  title,
  isDisabled,
  getInputProps,
  italicPlaceholder = true,
  fitContent = false,
}: DropdownInputProps) {
  return (
    <input
      {...getInputProps({
        placeholder,
        title,
        disabled: isDisabled,
        className: cn(
          "outline-none bg-transparent text-contrast not-italic",
          fitContent
            ? "w-auto field-sizing-content whitespace-nowrap text-sm"
            : "flex-1 min-w-0 text-ellipsis",
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
