import React, { useState, useMemo, useEffect } from "react";
import { useCombobox } from "downshift";
import { useTranslation } from "react-i18next";
import {
  gitProviderOptionKey,
  gitProviderOptionsMatch,
  type GitProviderOption,
} from "#/utils/git-provider-option";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { DropdownItem } from "../shared/dropdown-item";
import { GenericDropdownMenu } from "../shared/generic-dropdown-menu";
import { ToggleButton } from "../shared/toggle-button";
import { LoadingSpinner } from "../shared/loading-spinner";
import { ErrorMessage } from "../shared/error-message";
import { EmptyState } from "../shared/empty-state";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";

export interface GitProviderDropdownProps {
  options: GitProviderOption[];
  value?: GitProviderOption | null;
  placeholder?: string;
  className?: string;
  errorMessage?: string;
  disabled?: boolean;
  isLoading?: boolean;
  onChange?: (option: GitProviderOption | null) => void;
  inputClassName?: string;
  toggleButtonClassName?: string;
  itemClassName?: string;
}

export function GitProviderDropdown({
  options,
  value,
  placeholder,
  className,
  errorMessage,
  disabled = false,
  isLoading = false,
  onChange,
  inputClassName,
  toggleButtonClassName,
  itemClassName,
}: GitProviderDropdownProps) {
  const { t } = useTranslation("openhands");
  const [inputValue, setInputValue] = useState("");
  const [localSelectedItem, setLocalSelectedItem] =
    useState<GitProviderOption | null>(value || null);

  // Filter providers based on input value
  const filteredProviders = useMemo(() => {
    // If we have a selected provider and the input matches it exactly, show all providers
    if (inputValue === localSelectedItem?.label) {
      return options;
    }

    // If no input value, show all providers
    if (!inputValue?.trim()) {
      return options;
    }

    // Filter providers based on input
    return options.filter((option) =>
      option.label.toLowerCase().includes(inputValue.toLowerCase()),
    );
  }, [options, inputValue, localSelectedItem]);

  const {
    isOpen,
    getToggleButtonProps,
    getMenuProps,
    getInputProps,
    highlightedIndex,
    getItemProps,
    selectedItem,
  } = useCombobox({
    items: filteredProviders,
    itemToString: (item) => item?.label ?? "",
    selectedItem: localSelectedItem,
    onSelectedItemChange: ({ selectedItem: newSelectedItem }) => {
      setLocalSelectedItem(newSelectedItem || null);
      onChange?.(newSelectedItem || null);
    },
    onInputValueChange: ({ inputValue: newInputValue }) => {
      setInputValue(newInputValue || "");
    },
    inputValue,
  });

  // Sync with external value prop
  useEffect(() => {
    if (!gitProviderOptionsMatch(value, localSelectedItem)) {
      setLocalSelectedItem(value || null);
    }
  }, [value, localSelectedItem]);

  // Update input value when selection changes (but not when user is typing)
  useEffect(() => {
    if (selectedItem && !isOpen) {
      setInputValue(selectedItem.label);
    } else if (!selectedItem) {
      setInputValue("");
    }
  }, [selectedItem, isOpen]);

  const renderItem = (
    item: GitProviderOption,
    index: number,
    currentHighlightedIndex: number,
    currentSelectedItem: GitProviderOption | null,
    currentGetItemProps: any, // eslint-disable-line @typescript-eslint/no-explicit-any
  ) => (
    <DropdownItem
      key={gitProviderOptionKey(item)}
      item={item}
      index={index}
      isSelected={gitProviderOptionsMatch(item, currentSelectedItem)}
      getItemProps={currentGetItemProps}
      getDisplayText={(option) => option.label}
      getItemKey={gitProviderOptionKey}
      isProviderDropdown
      itemClassName={itemClassName}
    />
  );

  const renderEmptyState = (currentInputValue: string) => (
    <EmptyState
      inputValue={currentInputValue}
      searchMessage="No providers found"
      emptyMessage="No providers available"
      testId="git-provider-dropdown-empty"
    />
  );

  return (
    <div className={cn("relative", className)}>
      <div className="group relative text-muted hover:text-contrast">
        {/* Provider icon */}
        {selectedItem && (
          <div className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10">
            <GitProviderIcon
              gitProvider={selectedItem.provider}
              className="min-w-3.5 min-h-3.5 w-3.5 h-3.5"
            />
          </div>
        )}

        <input
          {...getInputProps({
            disabled,
            placeholder:
              placeholder ?? t(I18nKey.COMMON$SELECT_PROVIDER_PLACEHOLDER),
            readOnly: true, // Make it non-searchable like the original
            className: cn(
              "w-29.5 h-6 py-0 border border-border-input rounded shadow-none h-6 min-h-6 max-h-6 ",
              "text-inherit bg-tertiary placeholder:text-muted",
              "focus:outline-none focus:ring-0 focus:border-border-input",
              "disabled:bg-tertiary disabled:cursor-not-allowed disabled:opacity-60",
              "pl-1.5 pr-px cursor-pointer text-xs font-normal leading-5", // Space for toggle button and pointer cursor
              selectedItem && "pl-6",
              inputClassName,
            ),
          })}
          data-testid="git-provider-dropdown"
        />

        <div className="absolute right-0 top-1/2 transform -translate-y-1/2 flex items-center gap-1">
          <ToggleButton
            isOpen={isOpen}
            disabled={disabled}
            getToggleButtonProps={getToggleButtonProps}
            iconClassName={toggleButtonClassName}
          />
        </div>

        {isLoading && <LoadingSpinner hasSelection={!!selectedItem} />}
      </div>

      <GenericDropdownMenu
        isOpen={isOpen}
        filteredItems={filteredProviders}
        inputValue={inputValue}
        highlightedIndex={highlightedIndex}
        selectedItem={selectedItem}
        getMenuProps={getMenuProps}
        getItemProps={getItemProps}
        renderItem={renderItem}
        renderEmptyState={renderEmptyState}
        itemKey={gitProviderOptionKey}
      />

      <ErrorMessage isError={!!errorMessage} message={errorMessage} />
    </div>
  );
}
