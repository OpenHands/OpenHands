import React, {
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
} from "react";
import { useCombobox } from "downshift";
import { useTranslation } from "react-i18next";
import { Provider, ProviderOptions } from "#/types/settings";
import { GitRepository } from "#/types/git";
import { useDebounce } from "#/hooks/use-debounce";
import { cn } from "#/utils/utils";

import { ClearButton } from "../shared/clear-button";
import { ToggleButton } from "../shared/toggle-button";
import { ErrorMessage } from "../shared/error-message";
import { DropdownItem } from "../shared/dropdown-item";
import { EmptyState } from "../shared/empty-state";
import { useUrlSearch } from "./use-url-search";
import { useRepositoryData } from "./use-repository-data";
import { GenericDropdownMenu } from "../shared/generic-dropdown-menu";
import { useConfig } from "#/hooks/query/use-config";
import { I18nKey } from "#/i18n/declaration";
import RepoIcon from "#/icons/repo.svg?react";
import { useHomeStore } from "#/stores/home-store";
import { Typography } from "#/ui/typography";

export interface GitRepoDropdownProps {
  provider: Provider;
  value?: string | null;
  repositoryName?: string | null;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  onChange?: (repository?: GitRepository) => void;
}

export function GitRepoDropdown({
  provider,
  value,
  repositoryName,
  placeholder = "Search repositories...",
  className,
  disabled = false,
  onChange,
}: GitRepoDropdownProps) {
  const { t } = useTranslation();
  const { data: config } = useConfig();
  const { recentRepositories: storedRecentRepositories } = useHomeStore();
  const [inputValue, setInputValue] = useState("");
  const [localSelectedItem, setLocalSelectedItem] =
    useState<GitRepository | null>(null);
  const debouncedInputValue = useDebounce(inputValue, 300);
  const menuRef = useRef<HTMLUListElement>(null);

  // Process search input to handle URLs
  const processedSearchInput = useMemo(() => {
    if (debouncedInputValue.startsWith("https://")) {
      const match = debouncedInputValue.match(
        /https:\/\/[^/]+\/([^/]+\/[^/]+)/,
      );
      return match ? match[1] : debouncedInputValue;
    }
    return debouncedInputValue;
  }, [debouncedInputValue]);

  // URL search functionality
  const { urlSearchResults, isUrlSearchLoading } = useUrlSearch(
    inputValue,
    provider,
  );

  // Repository data management
  const {
    repositories,
    selectedRepository,
    fetchNextPage,
    hasNextPage,
    isLoading,
    isFetchingNextPage,
    isError,
    isSearchLoading,
  } = useRepositoryData(
    provider,
    disabled,
    processedSearchInput,
    urlSearchResults,
    inputValue,
    value,
    repositoryName,
  );

  /**
   * `selectedRepository` is derived from fetched/search results. A repo selected
   * from "Most Recent" may not be present in those results yet, so use the
   * locally selected item as a fallback to keep the UI stable.
   */
  const effectiveSelectedRepository = selectedRepository ?? localSelectedItem;

  // Get recent repositories filtered by provider and input keyword
  const recentRepositories = useMemo(() => {
    const allRecentRepos = storedRecentRepositories;
    const providerFilteredRepos = allRecentRepos.filter(
      (repo) => repo.git_provider === provider,
    );

    // If no input value, return all recent repos for this provider
    if (!inputValue?.trim()) {
      return providerFilteredRepos;
    }

    // Filter by input keyword
    const filterText = inputValue.startsWith("https://")
      ? processedSearchInput
      : inputValue;

    return providerFilteredRepos.filter((repo) =>
      repo.full_name.toLowerCase().includes(filterText.toLowerCase()),
    );
  }, [storedRecentRepositories, provider, inputValue, processedSearchInput]);

  // Helper function to prioritize recent repositories at the top
  const prioritizeRecentRepositories = useCallback(
    (repoList: GitRepository[]) => {
      const recentRepoIds = new Set(recentRepositories.map((repo) => repo.id));
      const otherRepos = repoList.filter((repo) => !recentRepoIds.has(repo.id));
      return [...recentRepositories, ...otherRepos];
    },
    [recentRepositories],
  );

  // Filter repositories based on input value
  const filteredRepositories = useMemo(() => {
    let baseRepositories: GitRepository[];

    // If we have URL search results, show them directly (no filtering needed)
    if (urlSearchResults.length > 0) {
      baseRepositories = repositories;
    }
    // If we have a selected repository and the input matches it exactly, show all repositories
    else if (
      selectedRepository &&
      inputValue === selectedRepository.full_name
    ) {
      baseRepositories = repositories;
    }
    // If no input value, show all repositories
    else if (!inputValue?.trim()) {
      baseRepositories = repositories;
    }
    // For URL inputs, use the processed search input for filtering
    else {
      const filterText = inputValue.startsWith("https://")
        ? processedSearchInput
        : inputValue;

      baseRepositories = repositories.filter((repo) =>
        repo.full_name.toLowerCase().includes(filterText.toLowerCase()),
      );
    }

    // Prioritize recent repositories at the top
    return prioritizeRecentRepositories(baseRepositories);
  }, [
    repositories,
    inputValue,
    selectedRepository,
    urlSearchResults,
    processedSearchInput,
    prioritizeRecentRepositories,
  ]);

  // Handle selection
  const handleSelectionChange = useCallback(
    (selectedItem: GitRepository | null) => {
      setLocalSelectedItem(selectedItem);
      onChange?.(selectedItem || undefined);
      // Update input value to show selected item
      if (selectedItem) {
        setInputValue(selectedItem.full_name);
      }
    },
    [onChange],
  );

  // Handle clear selection
  const handleClear = useCallback(() => {
    setLocalSelectedItem(null);
    handleSelectionChange(null);
    setInputValue("");
  }, [handleSelectionChange]);

  // Handle scroll to bottom for pagination
  const handleMenuScroll = useCallback(
    (event: React.UIEvent<HTMLUListElement>) => {
      const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
      const isNearBottom = scrollTop + clientHeight >= scrollHeight - 10;

      if (isNearBottom && hasNextPage && !isFetchingNextPage) {
        fetchNextPage();
      }
    },
    [hasNextPage, isFetchingNextPage, fetchNextPage],
  );

  const {
    isOpen,
    getToggleButtonProps,
    getMenuProps,
    getInputProps,
    highlightedIndex,
    getItemProps,
    selectedItem,
  } = useCombobox({
    items: filteredRepositories,
    itemToString: (item) => item?.full_name || "",
    selectedItem: localSelectedItem,
    onSelectedItemChange: ({ selectedItem: newSelectedItem }) => {
      handleSelectionChange(newSelectedItem);
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

  // Sync localSelectedItem with external value prop
  useEffect(() => {
    if (selectedRepository) {
      setLocalSelectedItem(selectedRepository);
    } else if (value === null) {
      setLocalSelectedItem(null);
    }
  }, [selectedRepository, value]);

  // Initialize input value when selectedRepository changes (but not when user is typing)
  useEffect(() => {
    if (isOpen) return;
    setInputValue(effectiveSelectedRepository?.full_name ?? "");
  }, [effectiveSelectedRepository, isOpen]);

  const isLoadingState =
    isLoading || isSearchLoading || isFetchingNextPage || isUrlSearchLoading;

  // Create sticky footer item for GitHub provider
  const stickyFooterItem = useMemo(() => {
    if (
      !config?.APP_SLUG ||
      provider !== ProviderOptions.github ||
      config.APP_MODE !== "saas"
    )
      return null;

    const githubHref = `https://github.com/apps/${config.APP_SLUG}/installations/new`;

    return (
      <a
        href={githubHref}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center w-full px-2 py-2 text-sm text-white hover:bg-[#5C5D62] rounded-md transition-colors duration-150 font-normal"
        onMouseDown={(e) => {
          // Prevent downshift from closing the menu when clicking the sticky footer
          e.preventDefault();
          e.stopPropagation();
        }}
      >
        {t(I18nKey.HOME$ADD_GITHUB_REPOS)}
      </a>
    );
  }, [provider, config, t]);

  const renderItem = (
    item: GitRepository,
    index: number,
    itemHighlightedIndex: number,
    itemSelectedItem: GitRepository | null,
    itemGetItemProps: any, // eslint-disable-line @typescript-eslint/no-explicit-any
  ) => (
    <DropdownItem
      key={item.id}
      item={item}
      index={index}
      isSelected={itemSelectedItem?.id === item.id}
      getItemProps={itemGetItemProps}
      getDisplayText={(repo) => repo.full_name}
      getItemKey={(repo) => repo.id}
    />
  );

  const renderEmptyState = (emptyInputValue: string) => (
    <EmptyState
      inputValue={emptyInputValue}
      searchMessage={t(I18nKey.HOME$NO_REPOSITORY_FOUND)}
      emptyMessage={t(I18nKey.COMMON$NO_REPOSITORY)}
      testId="git-repo-dropdown-empty"
    />
  );

  // Create sticky top item for recent repositories
  const stickyTopItem = useMemo(() => {
    if (recentRepositories.length === 0) {
      return null;
    }

    return (
      <div>
        <Typography.Text className="text-xs text-[#FAFAFA] font-semibold leading-4 pl-2">
          {t(I18nKey.COMMON$MOST_RECENT)}
        </Typography.Text>
      </div>
    );
  }, [recentRepositories, localSelectedItem, getItemProps, t]);

  return (
    <div className={cn("relative", className)}>
      <div className="relative">
        <div className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10">
          {isLoadingState ? (
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
          ) : (
            <RepoIcon width={16} height={16} />
          )}
        </div>
        <input
          // eslint-disable-next-line react/jsx-props-no-spreading
          {...getInputProps({
            disabled,
            placeholder,
            className: cn(
              "w-full px-3 py-2 border border-[#727987] rounded-sm shadow-none h-[42px] min-h-[42px] max-h-[42px]",
              "bg-[#454545] text-[#A3A3A3] placeholder:text-[#A3A3A3]",
              "focus:outline-none focus:ring-0 focus:border-[#727987]",
              "disabled:bg-[#363636] disabled:cursor-not-allowed disabled:opacity-60",
              "pl-7 pr-16 text-sm font-normal leading-5", // Space for clear and toggle buttons
            ),
            // Direct onChange for cursor position preservation
            onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
              setInputValue(e.target.value);
            },
          })}
          data-testid="git-repo-dropdown"
        />

        <div className="absolute right-1 top-1/2 transform -translate-y-1/2 flex items-center">
          {effectiveSelectedRepository && (
            <ClearButton disabled={disabled} onClear={handleClear} />
          )}

          <ToggleButton
            isOpen={isOpen}
            disabled={disabled}
            getToggleButtonProps={getToggleButtonProps}
            iconClassName="w-10 h-10"
          />
        </div>
      </div>

      <GenericDropdownMenu
        isOpen={isOpen}
        filteredItems={filteredRepositories}
        inputValue={inputValue}
        highlightedIndex={highlightedIndex}
        selectedItem={selectedItem}
        getMenuProps={getMenuProps}
        getItemProps={getItemProps}
        onScroll={handleMenuScroll}
        menuRef={menuRef}
        renderItem={renderItem}
        renderEmptyState={renderEmptyState}
        stickyTopItem={stickyTopItem}
        stickyFooterItem={stickyFooterItem}
        testId="git-repo-dropdown-menu"
        numberOfRecentItems={recentRepositories.length}
        itemKey={(repo) => repo.id}
      />

      <ErrorMessage isError={isError} />
    </div>
  );
}
