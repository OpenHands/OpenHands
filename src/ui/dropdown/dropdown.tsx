import React, { useState } from "react";
import { useCombobox } from "downshift";
import { cn } from "#/utils/utils";
import { DropdownOption } from "./types";
import { dropdownTriggerShellClassName } from "#/utils/dropdown-classes";
import { LoadingSpinner } from "./loading-spinner";
import { ClearButton } from "./clear-button";
import { ToggleButton } from "./toggle-button";
import { DropdownMenu } from "./dropdown-menu";
import { DropdownInput } from "./dropdown-input";

// Equivalent to Tailwind's `sr-only`, inlined so we don't depend on the
// utility class being preserved by the host project's CSS pipeline.
const visuallyHiddenStyle: React.CSSProperties = {
  position: "absolute",
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: "hidden",
  clip: "rect(0, 0, 0, 0)",
  whiteSpace: "nowrap",
  border: 0,
};

interface DropdownProps {
  options: DropdownOption[];
  emptyMessage?: string;
  clearable?: boolean;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  defaultValue?: DropdownOption;
  onChange?: (item: DropdownOption | null) => void;
  testId?: string;
  className?: string;
  footer?: React.ReactNode;
  openUpward?: boolean;
  hideTrigger?: boolean;
  defaultOpen?: boolean;
  /** Open the dropdown menu on hover instead of requiring a click. */
  openOnHover?: boolean;
  /** When false, the combobox placeholder uses normal (non-italic) type. */
  italicPlaceholder?: boolean;
  /** Size the trigger to its label instead of stretching to the container width. */
  fitContent?: boolean;
}

export function Dropdown({
  options,
  emptyMessage = "No options",
  clearable = false,
  loading = false,
  disabled = false,
  placeholder,
  defaultValue,
  onChange,
  testId,
  className,
  footer,
  openUpward = false,
  hideTrigger = false,
  defaultOpen = false,
  openOnHover = false,
  italicPlaceholder = true,
  fitContent = false,
}: DropdownProps) {
  const closeTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const [inputValue, setInputValue] = useState(defaultValue?.label ?? "");
  const [searchTerm, setSearchTerm] = useState("");

  React.useEffect(
    () => () => {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
        closeTimerRef.current = null;
      }
    },
    [],
  );

  const filteredOptions = options.filter((option) =>
    option.label.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  const {
    isOpen,
    selectedItem,
    selectItem,
    openMenu,
    closeMenu,
    getToggleButtonProps,
    getMenuProps,
    getItemProps,
    getInputProps,
  } = useCombobox({
    items: filteredOptions,
    itemToString: (item) => item?.label ?? "",
    inputValue,
    stateReducer: (state, actionAndChanges) =>
      actionAndChanges.type === useCombobox.stateChangeTypes.InputClick &&
      state.isOpen
        ? { ...actionAndChanges.changes, isOpen: true }
        : actionAndChanges.changes,
    initialIsOpen: defaultOpen,
    onInputValueChange: ({ inputValue: newValue }) => {
      setInputValue(newValue ?? "");
      setSearchTerm(newValue ?? "");
    },
    defaultSelectedItem: defaultValue,
    onSelectedItemChange: ({ selectedItem: newSelectedItem }) => {
      onChange?.(newSelectedItem ?? null);
    },
    onIsOpenChange: ({
      isOpen: newIsOpen,
      selectedItem: currentSelectedItem,
    }) => {
      if (newIsOpen) {
        // Clear the input on open so the user sees an empty search box
        // (with the placeholder reminding them of the current value)
        // and the full options list. Otherwise the active label would
        // appear both in the trigger AND as the highlighted menu row.
        setInputValue("");
        setSearchTerm("");
      } else {
        setInputValue(currentSelectedItem?.label ?? "");
        setSearchTerm("");
      }
    },
  });

  const isDisabled = loading || disabled;

  // `selectedItem` is downshift's internal state, frozen to whatever
  // initialized it. Resolve the currently selected option against the
  // live `options` array so per-option fields like `prefix` (e.g. a
  // status indicator that re-renders on a timer) update on the trigger
  // without remounting the dropdown.
  const liveSelectedOption = selectedItem
    ? (options.find((o) => o.value === selectedItem.value) ?? selectedItem)
    : null;

  // Wrap getInputProps to inject a direct onChange handler that preserves
  // cursor position. Downshift's default onInputValueChange resets cursor
  // to end of input on every keystroke; reading from e.target.value keeps
  // the browser's native cursor position intact.
  const getInputPropsWithCursorFix = (props?: object) =>
    getInputProps({
      ...props,
      onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputValue(e.target.value);
        setSearchTerm(e.target.value);
      },
    });

  return (
    <div
      className={cn("relative", fitContent ? "inline-block w-auto" : "w-full")}
      data-testid={testId}
      onMouseEnter={
        openOnHover
          ? () => {
              if (closeTimerRef.current) {
                clearTimeout(closeTimerRef.current);
                closeTimerRef.current = null;
              }
              openMenu();
            }
          : undefined
      }
      onMouseLeave={
        openOnHover
          ? () => {
              if (closeTimerRef.current) {
                clearTimeout(closeTimerRef.current);
              }
              closeTimerRef.current = setTimeout(() => closeMenu(), 150);
            }
          : undefined
      }
    >
      {!hideTrigger ? (
        <div
          className={cn(
            dropdownTriggerShellClassName,
            fitContent ? "w-auto" : "w-full",
            isDisabled && "cursor-not-allowed opacity-60",
            className,
          )}
        >
          {liveSelectedOption?.prefix ? (
            <span className="flex items-center shrink-0">
              {liveSelectedOption.prefix}
            </span>
          ) : null}
          <DropdownInput
            placeholder={placeholder}
            isDisabled={isDisabled}
            getInputProps={getInputPropsWithCursorFix}
            italicPlaceholder={italicPlaceholder}
            fitContent={fitContent}
          />
          {loading && <LoadingSpinner />}
          {clearable && selectedItem && (
            <ClearButton onClear={() => selectItem(null)} />
          )}
          <ToggleButton
            isOpen={isOpen}
            isDisabled={isDisabled}
            getToggleButtonProps={getToggleButtonProps}
          />
        </div>
      ) : (
        // downshift's useCombobox always expects getInputProps() (and the
        // toggle button) to be wired up. When the trigger is hidden (e.g.
        // collapsed-sidebar popover) we still need to mount a real input
        // so it stops warning every render. Keep it visually hidden but
        // present in the DOM for accessibility.
        <>
          <input
            {...getInputPropsWithCursorFix({
              // eslint-disable-next-line i18next/no-literal-string -- i18n-free UI primitive; callers supply translated labels
              "aria-label": placeholder ?? "Filter options",
              tabIndex: -1,
            })}
            style={visuallyHiddenStyle}
          />
          <button
            type="button"
            {...getToggleButtonProps({ tabIndex: -1 })}
            style={visuallyHiddenStyle}
            aria-hidden
          />
        </>
      )}
      <DropdownMenu
        isOpen={isOpen}
        filteredOptions={filteredOptions}
        selectedItem={selectedItem}
        emptyMessage={emptyMessage}
        getMenuProps={getMenuProps}
        getItemProps={getItemProps}
        footer={footer}
        openUpward={openUpward}
        fitContent={fitContent}
      />
    </div>
  );
}
