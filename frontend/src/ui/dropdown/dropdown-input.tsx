/* eslint-disable react/jsx-props-no-spreading */
import { cn } from "#/utils/utils";

interface DropdownInputProps {
  placeholder?: string;
  isDisabled: boolean;
  getInputProps: (props?: object) => object;
}

export function DropdownInput({
  placeholder,
  isDisabled,
  getInputProps,
}: DropdownInputProps) {
  return (
    <input
      {...getInputProps({
        placeholder,
        disabled: isDisabled,
        className: cn(
          "flex-grow outline-none bg-transparent text-white",
          "placeholder:italic placeholder:text-tertiary-alt",
        ),
      })}
    />
  );
}
