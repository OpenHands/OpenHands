import React, { useState, useMemo, useEffect } from "react";
import { useCombobox } from "downshift";
import { Provider } from "#/types/settings";
import { cn } from "#/utils/utils";
import { DropdownItem } from "../shared/dropdown-item";
import { GenericDropdownMenu } from "../shared/generic-dropdown-menu";
import { ToggleButton } from "../shared/toggle-button";
import { LoadingSpinner } from "../shared/loading-spinner";
import { ErrorMessage } from "../shared/error-message";
import { EmptyState } from "../shared/empty-state";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";

export interface GitProviderDropdownProps {
  providers: Provider[];
  value?: Provider | null;
  placeholder?: string;
  className?: string;
  errorMessage?: string;
  disabled?: boolean;
  isLoading?: boolean;
  onChange?: (provider: Provider | null) => void;
  inputClassName?: string;
  toggleButtonClassName?: string;
  itemClassName?: string;
}

export function GitProviderDropdown({
  providers,
  value,
  placeholder = "Select Provider",
  className,
  errorMessage,
  disabled = false,
  isLoading = false,
  onChange,
  inputClassName,
  toggleButtonClassName,
  itemClassName,
}: GitProviderDropdownProps) {
  const [inputValue, setInputValue] = useState("");
  const [localSelectedItem, setLocalSelectedItem] = useState<Provider | null>(
    value || null,
  );

  // Format provider names for display
  const formatProviderName = (provider: Provider): string => {
    switch (provider) {
      case "github":
        return "GitHub";
      case "gitlab":
        return "GitLab";
      case "bitbucket":
        return "Bitbucket";
      case "azure_devops":
        return "Azure DevOps";
      case "enterprise_sso":
        return "Enterprise SSO";
      default:
        // Fallback for any future provider types
        return (
          (provider as string).charAt(0).toUpperCase() +
          (provider as string).slice(1)
        );
    }
  };

  // Filter providers based on input value
  const filteredProviders = useMemo(() => {
    // If we have a selected provider and the input matches it exactly, show all providers
    if (
      localSelectedItem &&
      inputValue === formatProviderName(localSelectedItem)
    ) {
      return providers;
    }

    // If no input value, show all providers
    if (!inputValue?.trim()) {
      return providers;
    }

    // Filter providers based on input
    return providers.filter((provider) =>
      formatProviderName(provider)
        .toLowerCase()
        .includes(inputValue.toLowerCase()),
    );
  }, [providers, inputValue, localSelectedItem]);

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
    itemToString: (item) => (item ? formatProviderName(item) : ""),
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
    if (value !== localSelectedItem) {
      setLocalSelectedItem(value || null);
    }
  }, [value, localSelectedItem]);

  // Update input value when selection changes (but not when user is typing)
  useEffect(() => {
    if (selectedItem && !isOpen) {
      setInputValue(formatProviderName(selectedItem));
    } else if (!selectedItem) {
      setInputValue("");
    }
  }, [selectedItem, isOpen]);

  const renderItem = (
    item: Provider,
    index: number,
    currentHighlightedIndex: number,
    currentSelectedItem: Provider | null,
    currentGetItemProps: any, // eslint-disable-line @typescript-eslint/no-explicit-any
  ) => (
    <DropdownItem
      key={item}
      item={item}
      index={index}
      isSelected={item === currentSelectedItem}
      getItemProps={currentGetItemProps}
      getDisplayText={formatProviderName}
      getItemKey={(provider) => provider}
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
      <div className="relative">
        {/* Provider icon */}
        {selectedItem && (
          <div className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10">
            <GitProviderIcon
              gitProvider={selectedItem}
              className="min-w-[14px] min-h-[14px] w-[14px] h-[14px]"
            />
          </div>
        )}

        <input
          // eslint-disable-next-line react/jsx-props-no-spreading
          {...getInputProps({
            disabled,
            placeholder,
            readOnly: true, // Make it non-searchable like the original
            className: cn(
              "w-29.5 h-6 py-0 border border-[#727987] rounded shadow-none h-6 min-h-6 max-h-6 ",
              "bg-[#454545] text-[#A3A3A3] placeholder:text-[#A3A3A3]",
              "focus:outline-none focus:ring-0 focus:border-[#727987]",
              "disabled:bg-[#363636] disabled:cursor-not-allowed disabled:opacity-60",
              "pl-1.5 pr-[1px] cursor-pointer text-xs font-normal leading-5", // Space for toggle button and pointer cursor
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
            iconClassName={cn(
              "w-[23px] h-[23px] translate-y-[1px]",
              toggleButtonClassName,
            )}
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
        itemKey={(provider) => provider}
      />

      <ErrorMessage isError={!!errorMessage} message={errorMessage} />
    </div>
  );
}
