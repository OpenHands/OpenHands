import React, {
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
} from "react";
import { useCombobox } from "downshift";
import { useTranslation } from "react-i18next";
import { Branch } from "#/types/git";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { useDebounce } from "#/hooks/use-debounce";
import { cn } from "#/utils/utils";
import { formControlFieldClassName } from "#/utils/form-control-classes";
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
  placeholder,
  disabled = false,
  className,
}: GitBranchDropdownProps) {
  const { t } = useTranslation("openhands");
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
    inputValue,
    // Override Downshift's default input-click behavior to avoid closing/reopening
    // the menu, which would reset scroll position and break search continuity.
    stateReducer: (state, actionAndChanges) =>
      actionAndChanges.type === useCombobox.stateChangeTypes.InputClick &&
      state.isOpen
        ? { ...actionAndChanges.changes, isOpen: true }
        : actionAndChanges.changes,
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
    if (selectedBranch && !isOpen) {
      setInputValue(selectedBranch.name);
    } else if (!selectedBranch && !isOpen) {
      setInputValue("");
    }
  }, [selectedBranch, isOpen]);

  const isLoadingState = isLoading || isSearchLoading || isFetchingNextPage;

  return (
    <div className={cn("relative", className)}>
      <div className="group relative text-[var(--oh-muted)] hover:text-white">
        <div className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10">
          {isLoadingState ? (
            <div className="animate-spin h-4 w-4 border-2 border-transparent border-t-white rounded-full" />
          ) : (
            <BranchIcon width={16} height={16} />
          )}
        </div>
        <input
          {...getInputProps({
            disabled: disabled || !repository,
            placeholder:
              placeholder ?? t(I18nKey.COMMON$SELECT_BRANCH_PLACEHOLDER),
            className: cn(
              formControlFieldClassName,
              "text-inherit shadow-none pl-7 pr-16 text-sm font-normal leading-5",
              "placeholder:text-[var(--oh-muted)]",
              "disabled:cursor-not-allowed disabled:opacity-60",
            ),
            // Direct onChange for cursor position preservation
            onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
              setInputValue(e.target.value);
            },
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
