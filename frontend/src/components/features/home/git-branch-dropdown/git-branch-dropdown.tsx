import React, {
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
} from "react";
import { useCombobox } from "downshift";
import { Branch } from "#/types/git";
import { Provider } from "#/types/settings";
import { useDebounce } from "#/hooks/use-debounce";
import { cn } from "#/utils/utils";
import { useBranchData } from "#/hooks/query/use-branch-data";

import { ClearButton } from "../shared/clear-button";
import { ToggleButton } from "../shared/toggle-button";
import { ErrorMessage } from "../shared/error-message";
import { BranchDropdownMenu } from "./branch-dropdown-menu";
import BranchIcon from "#/icons/u-code-branch.svg?react";

export interface GitBranchDropdownProps {
  repository: string | null;
  provider: Provider;
  selectedBranch: Branch | null;
  onBranchSelect: (branch: Branch | null) => void;
  defaultBranch?: string | null;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export function GitBranchDropdown({
  repository,
  provider,
  selectedBranch,
  onBranchSelect,
  defaultBranch,
  placeholder = "Select branch...",
  disabled = false,
  className,
}: GitBranchDropdownProps) {
  const [inputValue, setInputValue] = useState("");
  const [userManuallyCleared, setUserManuallyCleared] = useState(false);
  const debouncedInputValue = useDebounce(inputValue, 300);
  const menuRef = useRef<HTMLUListElement>(null);

  // Process search input (debounced and filtered)
  const processedSearchInput = useMemo(
    () =>
      debouncedInputValue.trim().length > 0 ? debouncedInputValue.trim() : "",
    [debouncedInputValue],
  );

  // Use the new branch data hook with default branch prioritization
  const {
    branches: filteredBranches,
    isLoading,
    isError,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isSearchLoading,
  } = useBranchData(
    repository,
    provider,
    defaultBranch || null,
    processedSearchInput,
    inputValue,
    selectedBranch,
  );

  const error = isError ? new Error("Failed to load branches") : null;

  // Handle clear
  const handleClear = useCallback(() => {
    setInputValue("");
    onBranchSelect(null);
    setUserManuallyCleared(true); // Mark that user manually cleared the branch
  }, [onBranchSelect]);

  // Handle branch selection
  const handleBranchSelect = useCallback(
    (branch: Branch | null) => {
      onBranchSelect(branch);
      setInputValue("");
    },
    [onBranchSelect],
  );

  // Handle input value change
  const handleInputValueChange = useCallback(
    ({ inputValue: newInputValue }: { inputValue?: string }) => {
      if (newInputValue !== undefined) {
        setInputValue(newInputValue);
      }
    },
    [],
  );

  // Handle menu scroll for infinite loading
  const handleMenuScroll = useCallback(
    (event: React.UIEvent<HTMLUListElement>) => {
      const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
      if (
        scrollHeight - scrollTop <= clientHeight * 1.5 &&
        hasNextPage &&
        !isFetchingNextPage
      ) {
        fetchNextPage();
      }
    },
    [hasNextPage, isFetchingNextPage, fetchNextPage],
  );

  // Downshift configuration
  const {
    isOpen,
    selectedItem,
    highlightedIndex,
    getInputProps,
    getItemProps,
    getMenuProps,
    getToggleButtonProps,
  } = useCombobox({
    items: filteredBranches,
    selectedItem: selectedBranch,
    itemToString: (item) => item?.name || "",
    onSelectedItemChange: ({ selectedItem: newSelectedItem }) => {
      handleBranchSelect(newSelectedItem || null);
    },
    onInputValueChange: handleInputValueChange,
    inputValue,
  });

  // Reset branch selection when repository changes
  useEffect(() => {
    if (repository) {
      onBranchSelect(null);
      setUserManuallyCleared(false); // Reset the manual clear flag when repository changes
    }
  }, [repository, onBranchSelect]);

  // Auto-select default branch when branches are loaded and no branch is selected
  // But only if the user hasn't manually cleared the branch
  useEffect(() => {
    if (
      repository &&
      defaultBranch &&
      !selectedBranch &&
      !userManuallyCleared && // Don't auto-select if user manually cleared
      filteredBranches.length > 0 &&
      !isLoading
    ) {
      const defaultBranchObj = filteredBranches.find(
        (branch) => branch.name === defaultBranch,
      );

      if (defaultBranchObj) {
        onBranchSelect(defaultBranchObj);
      }
    }
  }, [
    repository,
    defaultBranch,
    selectedBranch,
    userManuallyCleared,
    filteredBranches,
    onBranchSelect,
    isLoading,
  ]);

  // Reset input when repository changes
  useEffect(() => {
    setInputValue("");
  }, [repository]);

  // Initialize input value when selectedBranch changes (but not when user is typing)
  useEffect(() => {
    if (selectedBranch && !isOpen && inputValue !== selectedBranch.name) {
      setInputValue(selectedBranch.name);
    } else if (!selectedBranch && !isOpen && inputValue) {
      setInputValue("");
    }
  }, [selectedBranch, isOpen, inputValue]);

  const isLoadingState = isLoading || isSearchLoading || isFetchingNextPage;

  return (
    <div className={cn("relative", className)}>
      <div className="relative">
        <div className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10">
          {isLoadingState ? (
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
          ) : (
            <BranchIcon width={16} height={16} />
          )}
        </div>
        <input
          // eslint-disable-next-line react/jsx-props-no-spreading
          {...getInputProps({
            disabled: disabled || !repository,
            placeholder,
            className: cn(
              "w-full px-3 py-2 border border-[#727987] rounded-sm shadow-none h-[42px] min-h-[42px] max-h-[42px]",
              "bg-[#454545] text-[#A3A3A3] placeholder:text-[#A3A3A3]",
              "focus:outline-none focus:ring-0 focus:border-[#727987]",
              "disabled:bg-[#363636] disabled:cursor-not-allowed disabled:opacity-60",
              "pl-7 pr-16 text-sm font-normal leading-5", // Space for clear and toggle buttons
            ),
          })}
          data-testid="git-branch-dropdown-input"
        />

        <div className="absolute right-1 top-1/2 transform -translate-y-1/2 flex items-center">
          {selectedBranch && (
            <ClearButton disabled={disabled} onClear={handleClear} />
          )}

          <ToggleButton
            isOpen={isOpen}
            disabled={disabled || !repository}
            getToggleButtonProps={getToggleButtonProps}
            iconClassName="w-10 h-10"
          />
        </div>
      </div>

      <BranchDropdownMenu
        isOpen={isOpen}
        filteredBranches={filteredBranches}
        inputValue={inputValue}
        highlightedIndex={highlightedIndex}
        selectedItem={selectedItem}
        getMenuProps={getMenuProps}
        getItemProps={getItemProps}
        onScroll={handleMenuScroll}
        menuRef={menuRef}
      />

      <ErrorMessage isError={!!error} />
    </div>
  );
}
