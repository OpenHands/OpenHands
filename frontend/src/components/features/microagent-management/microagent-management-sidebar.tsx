import { useEffect, useState, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Spinner } from "@heroui/react";
import { MicroagentManagementSidebarHeader } from "./microagent-management-sidebar-header";
import { MicroagentManagementSidebarTabs } from "./microagent-management-sidebar-tabs";
import { useGitRepositories } from "#/hooks/query/use-git-repositories";
import { useSearchRepositories } from "#/hooks/query/use-search-repositories";
import { GitProviderDropdown } from "#/components/features/home/git-provider-dropdown";
import { useMicroagentManagementStore } from "#/state/microagent-management-store";
import { GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";
import {
  cn,
  shouldIncludeRepository,
  getOpenHandsQuery,
  hasOpenHandsSuffix,
} from "#/utils/utils";
import { sanitizeQuery } from "#/utils/sanitize-query";
import { I18nKey } from "#/i18n/declaration";
import { useDebounce } from "#/hooks/use-debounce";

interface MicroagentManagementSidebarProps {
  isSmallerScreen?: boolean;
  providers: Provider[];
}

export function MicroagentManagementSidebar({
  isSmallerScreen = false,
  providers,
}: MicroagentManagementSidebarProps) {
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(
    providers.length > 0 ? providers[0] : null,
  );

  const [searchQuery, setSearchQuery] = useState("");
  const debouncedSearchQuery = useDebounce(searchQuery, 300);

  const {
    setPersonalRepositories,
    setOrganizationRepositories,
    setRepositories,
  } = useMicroagentManagementStore();

  const { t } = useTranslation();

  // Use Git repositories hook with pagination for infinite scrolling
  const {
    data: repositories,
    fetchNextPage,
    hasNextPage,
    isLoading,
    isFetchingNextPage,
  } = useGitRepositories({
    provider: selectedProvider,
    pageSize: 30, // Load 30 repositories per page
    enabled: !!selectedProvider,
  });

  // Server-side search functionality
  const { data: searchResults, isLoading: isSearchLoading } =
    useSearchRepositories(debouncedSearchQuery, selectedProvider, false, 500); // Increase page size to 500 to to retrieve all search results. This should be optimized in the future.

  const {
    data: userAndOrgLevelRepositorySearchResults,
    isLoading: isUserAndOrgLevelRepositoryLoading,
  } = useSearchRepositories(
    getOpenHandsQuery(selectedProvider),
    selectedProvider,
    false,
    500,
  );

  // Auto-select provider if there's only one
  useEffect(() => {
    if (providers.length > 0 && !selectedProvider) {
      setSelectedProvider(providers[0]);
    }
  }, [providers, selectedProvider]);

  const handleProviderChange = (provider: Provider | null) => {
    setSelectedProvider(provider);
    setSearchQuery("");
  };

  // Helper function to filter repositories by search query
  const filterRepositoriesByQuery = (
    inputRepositories: GitRepository[],
    query: string,
  ): GitRepository[] => {
    if (!query.trim()) {
      return inputRepositories;
    }

    const sanitizedQuery = sanitizeQuery(query);
    return inputRepositories.filter((repository: GitRepository) => {
      const sanitizedRepoName = sanitizeQuery(repository.full_name);
      return sanitizedRepoName.includes(sanitizedQuery);
    });
  };

  // Filter repositories based on search query and available data
  const filteredRepositories = useMemo(() => {
    // If we have search results, apply client-side filtering for exact matches
    if (debouncedSearchQuery && searchResults && searchResults.length > 0) {
      return filterRepositoriesByQuery(searchResults, debouncedSearchQuery);
    }

    // If no search query or no search results, use paginated repositories
    if (!repositories?.pages) return [];

    // Flatten all pages to get all repositories
    const allRepositories = repositories.pages.flatMap((page) => page.data);

    // Apply filtering to paginated repositories
    return filterRepositoriesByQuery(allRepositories, debouncedSearchQuery);
  }, [repositories, debouncedSearchQuery, searchResults]);

  // Process personal and organization repositories from search results
  useEffect(() => {
    if (!userAndOrgLevelRepositorySearchResults?.length) {
      setPersonalRepositories([]);
      setOrganizationRepositories([]);
      return;
    }

    const personalRepos: GitRepository[] = [];
    const organizationRepos: GitRepository[] = [];

    // Process personal repositories with exact match filtering
    if (userAndOrgLevelRepositorySearchResults?.length) {
      userAndOrgLevelRepositorySearchResults.forEach((repo: GitRepository) => {
        if (
          hasOpenHandsSuffix(repo, selectedProvider) &&
          shouldIncludeRepository(repo, debouncedSearchQuery)
        ) {
          if (repo.owner_type === "user") {
            personalRepos.push(repo);
          } else if (repo.owner_type === "organization") {
            organizationRepos.push(repo);
          }
        }
      });
    }

    setPersonalRepositories(personalRepos);
    setOrganizationRepositories(organizationRepos);
  }, [
    userAndOrgLevelRepositorySearchResults,
    selectedProvider,
    debouncedSearchQuery,
    setPersonalRepositories,
    setOrganizationRepositories,
  ]);

  // Process other repositories (non-OpenHands repositories) from filteredRepositories
  useEffect(() => {
    if (!filteredRepositories?.length) {
      setRepositories([]);
      return;
    }

    const otherRepos: GitRepository[] = [];

    filteredRepositories.forEach((repo: GitRepository) => {
      // Only include repositories that don't have the OpenHands suffix
      if (!hasOpenHandsSuffix(repo, selectedProvider)) {
        otherRepos.push(repo);
      }
    });

    setRepositories(otherRepos);
  }, [filteredRepositories, selectedProvider, setRepositories]);

  // Handle scroll to bottom for pagination
  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    // Only enable pagination when not searching
    if (debouncedSearchQuery && searchResults) {
      return;
    }

    const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - 10;

    if (isNearBottom && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  };

  return (
    <div
      className={cn(
        "w-[418px] h-full max-h-full overflow-y-auto overflow-x-hidden border-r border-[#525252] bg-[#24272E] rounded-tl-lg rounded-bl-lg py-10 px-6 flex flex-col custom-scrollbar-always",
        isSmallerScreen && "w-full border-none",
      )}
      onScroll={handleScroll}
    >
      <MicroagentManagementSidebarHeader />

      {/* Provider Selection */}
      {providers.length > 1 && (
        <div className="mt-6">
          <GitProviderDropdown
            providers={providers}
            value={selectedProvider}
            placeholder="Select Provider"
            onChange={handleProviderChange}
            className="w-full"
            inputClassName="w-full h-10 min-h-10 max-h-10 text-sm placeholder:text-tertiary-alt pl-6.5"
            toggleButtonClassName="w-10 h-10 translate-y-[1px]"
            itemClassName="text-sm"
          />
        </div>
      )}

      {/* Search Input */}
      <div className="flex flex-col gap-2 w-full mt-6">
        <label htmlFor="repository-search" className="sr-only">
          {t(I18nKey.COMMON$SEARCH_REPOSITORIES)}
        </label>
        <div className="relative">
          <input
            id="repository-search"
            name="repository-search"
            type="text"
            placeholder={`${t(I18nKey.COMMON$SEARCH_REPOSITORIES)}...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className={cn(
              "bg-tertiary border border-[#717888] bg-[#454545] w-full rounded-sm p-2 placeholder:text-tertiary-alt",
              "disabled:bg-[#2D2F36] disabled:border-[#2D2F36] disabled:cursor-not-allowed h-10 box-shadow-none outline-none",
              "pr-10", // Space for spinner
            )}
          />
          {isSearchLoading && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <Spinner size="sm" />
            </div>
          )}
        </div>
      </div>

      {isLoading || isUserAndOrgLevelRepositoryLoading ? (
        <div className="flex flex-col items-center justify-center gap-4 flex-1">
          <Spinner size="sm" />
          <span className="text-sm text-white">
            {t("HOME$LOADING_REPOSITORIES")}
          </span>
        </div>
      ) : (
        <>
          <MicroagentManagementSidebarTabs isSearchLoading={isSearchLoading} />

          {/* Show loading indicator for pagination (only when not searching) */}
          {isFetchingNextPage && !debouncedSearchQuery && (
            <div className="flex justify-center pt-2">
              <Spinner size="sm" />
              <span className="text-sm text-white ml-2">
                {t("HOME$LOADING_MORE_REPOSITORIES")}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
