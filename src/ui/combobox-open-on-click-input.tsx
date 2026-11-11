import { Input } from "@heroui/react";
import React, { useContext } from "react";
import { ComboBoxStateContext } from "react-aria-components";

interface ComboBoxOpenOnClickInputProps {
  testId: string;
  name: string;
  placeholder?: string;
  onFocus?: React.FocusEventHandler<HTMLInputElement>;
}

/**
 * The text input of a HeroUI ComboBox that opens its listbox on click, not only
 * on typing. While the listbox is open it owns Escape: React Aria closes the
 * listbox and lets Escape propagate, which would also close a dialog around
 * the field.
 */
export function ComboBoxOpenOnClickInput({
  testId,
  name,
  placeholder,
  onFocus,
}: ComboBoxOpenOnClickInputProps) {
  const state = useContext(ComboBoxStateContext);
  return (
    <Input
      data-testid={testId}
      name={name}
      className="border-0 bg-transparent shadow-none outline-none focus:ring-0 focus-visible:ring-0 data-[focused=true]:ring-0"
      placeholder={placeholder}
      onFocus={onFocus}
      onClick={() => {
        if (state && !state.isOpen) state.open(null, "manual");
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape" && state?.isOpen) event.stopPropagation();
      }}
    />
  );
}
