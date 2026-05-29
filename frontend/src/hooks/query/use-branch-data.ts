import { useMemo } from "react";
import { useRepositoryBranchesPaginated } from "./use-repository-branches";
import { useSearchBranches } from "./use-search-branches";
import { Branch } from "#/types/git";
import { Provider } from "#/types/settings";

export function useBranchData(
  repository: string | null,
  provider: Provider,
  defaultBranch: string | null,
  processedSearchInput: string,
  inputValue: string,
  selectedBranch?: Branch | null,
) {
  // Fetch branches with pagination
  const {
    data: branchData,
    fetchNextPage: fetchNextListPage,
    hasNextPage: hasNextListPage,
    isLoading: isListLoading,
    isFetchingNextPage: isFetchingNextListPage,
    isError: isListError,
  } = useRepositoryBranchesPaginated(repository, 30, provider);

  // Search branches when user types
  const {
    data: searchData,
    fetchNextPage: fetchNextSearchPage,
    hasNextPage: hasNextSearchPage,
    isLoading: isSearchLoading,
    isFetchingNextPage: isFetchingNextSearchPage,
    isError: isSearchError,
  } = useSearchBranches(repository, processedSearchInput, 30, provider);

  // Combine all branches from paginated data - use .items for V1 response
  const allBranches = useMemo(
    () =>
      branchData?.pages?.flatMap((page: { items: Branch[] }) => page.items) ||
      [],
    [branchData],
  );

  // Check if default branch is in the loaded branches
  const defaultBranchInLoaded = useMemo(
    () =>
      defaultBranch
        ? allBranches.find((branch: Branch) => branch.name === defaultBranch)
        : null,
    [allBranches, defaultBranch],
  );

  // Only search for default branch if it's not already in the loaded branches
  // and we have loaded some branches (to avoid searching immediately on mount)
  const shouldSearchDefaultBranch =
    defaultBranch &&
    !defaultBranchInLoaded &&
    allBranches.length > 0 &&
    !processedSearchInput; // Don't search for default branch when user is searching

  const { data: defaultBranchData, isLoading: isDefaultBranchLoading } =
    useSearchBranches(
      repository,
      shouldSearchDefaultBranch ? defaultBranch : "",
      30,
      provider,
    );
    const shouldUseSearch = useMemo(
      () =>
        Boolean(
          processedSearchInput &&
            searchData &&
            !(selectedBranch && inputValue === selectedBranch.name),
        ),
      [processedSearchInput, searchData, selectedBranch, inputValue],
    );

  // Get branches to display with default branch prioritized
  const branches = useMemo(() => {
    let branchesToUse = shouldUseSearch ? searchData : allBranches;

    // If we have a default branch, ensure it's at the top of the list
    if (defaultBranch) {
      // Use the already computed defaultBranchInLoaded or check in current branches
      let defaultBranchObj = shouldUseSearch
        ? branchesToUse.find((branch: Branch) => branch.name === defaultBranch)
        : defaultBranchInLoaded;

      // If not found in current branches, check if we have it from the default branch search
      if (
        !defaultBranchObj &&
        defaultBranchData &&
        defaultBranchData.length > 0
      ) {
        defaultBranchObj = defaultBranchData.find(
          (branch) => branch.name === defaultBranch,
        );

        // Add the default branch to the beginning of the list
        if (defaultBranchObj) {
          branchesToUse = [defaultBranchObj, ...branchesToUse];
        }
      } else if (defaultBranchObj) {
        // If found in current branches, move it to the front
        const otherBranches = branchesToUse.filter(
          (branch) => branch.name !== defaultBranch,
        );
        branchesToUse = [defaultBranchObj, ...otherBranches];
      }
    }

    return branchesToUse;
  }, [
    shouldUseSearch,
    searchData,
    allBranches,
    defaultBranch,
    defaultBranchInLoaded,
    defaultBranchData,
  ]);

  return {
    branches,
    allBranches,
    fetchNextPage: shouldUseSearch ? fetchNextSearchPage : fetchNextListPage,
    hasNextPage: shouldUseSearch ? hasNextSearchPage : hasNextListPage,
    isLoading: shouldUseSearch
      ? isSearchLoading
      : isListLoading || isDefaultBranchLoading,
    isFetchingNextPage: shouldUseSearch
      ? isFetchingNextSearchPage
      : isFetchingNextListPage,
    isError: shouldUseSearch ? isSearchError : isListError,
    isSearchLoading,
  };
}
