import React from "react";
import { useTranslation } from "react-i18next";

import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useNavigation } from "#/context/navigation-context";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { useGitRepositories } from "#/hooks/query/use-git-repositories";
import { useSearchRepositories } from "#/hooks/query/use-search-repositories";
import { useUserProviders } from "#/hooks/use-user-providers";
import { useDebounce } from "#/hooks/use-debounce";
import { useHomeStore } from "#/stores/home-store";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  dropdownMenuRowClassName,
  dropdownInstantColorClassName,
  dropdownMenuListClassName,
  dropdownMenuRowIconWrapperClassName,
} from "#/utils/dropdown-classes";
import { GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";
import RepoIcon from "#/icons/repo.svg?react";
import SearchIcon from "#/icons/search.svg?react";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { Divider } from "#/ui/divider";
import { NEW_CONVERSATION_DROPDOWN_SURFACE } from "./new-conversation-dropdown-styles";
import { usePopoverFixedPlacement } from "#/hooks/use-popover-fixed-placement";

export type CloudNewConversationMenuTriggerProps = {
  onClick: () => void;
  "aria-expanded": boolean;
  "aria-haspopup": "menu";
  disabled?: boolean;
};

export interface CloudNewConversationMenuProps {
  trigger: (props: CloudNewConversationMenuTriggerProps) => React.ReactNode;
  className?: string;
  popoverClassName: string;
  popoverTestId?: string;
  useFixedPlacement?: boolean;
}

interface RepoListItemProps {
  repo: GitRepository;
  disabled: boolean;
  onSelect: (repo: GitRepository) => void;
  itemClass: string;
}

function RepoListItem({
  repo,
  disabled,
  onSelect,
  itemClass,
}: RepoListItemProps) {
  return (
    <li>
      <button
        type="button"
        disabled={disabled}
        data-testid="launch-repository"
        data-repo-name={repo.full_name}
        onClick={() => onSelect(repo)}
        className={itemClass}
      >
        <span className={dropdownMenuRowIconWrapperClassName} aria-hidden>
          <RepoIcon width={14} height={14} />
        </span>
        <span className="truncate">{repo.full_name}</span>
      </button>
    </li>
  );
}

/**
 * Repository search + launch flow for cloud backends.
 * Shared by the sidebar "+ New conversation" control and the conversation
 * panel "new thread folder" opener.
 */
export function CloudNewConversationMenu({
  trigger,
  className,
  popoverClassName,
  popoverTestId = "new-conversation-popover",
  useFixedPlacement = false,
}: CloudNewConversationMenuProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();

  const { providers } = useUserProviders();
  const { lastSelectedProvider, setLastSelectedProvider } = useHomeStore();

  const [open, setOpen] = React.useState(false);
  const popoverRef = React.useRef<HTMLDivElement>(null);
  const triggerWrapRef = React.useRef<HTMLSpanElement>(null);
  const fixedBox = usePopoverFixedPlacement(triggerWrapRef, {
    open,
    enabled: useFixedPlacement,
  });
  const [selectedProvider, setSelectedProvider] =
    React.useState<Provider | null>(lastSelectedProvider ?? null);
  const [query, setQuery] = React.useState("");
  const debouncedQuery = useDebounce(query, 300);

  React.useEffect(() => {
    if (providers.length === 0) {
      if (selectedProvider !== null) setSelectedProvider(null);
      return;
    }
    if (selectedProvider && providers.includes(selectedProvider)) return;
    const fallback =
      lastSelectedProvider && providers.includes(lastSelectedProvider)
        ? lastSelectedProvider
        : providers[0];
    setSelectedProvider(fallback);
  }, [providers, selectedProvider, lastSelectedProvider]);

  const {
    data: repoPages,
    isLoading,
    isError,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useGitRepositories({ provider: selectedProvider });

  const { data: searchResults, isLoading: isSearchLoading } =
    useSearchRepositories(debouncedQuery, selectedProvider);

  const allRepositories = React.useMemo(
    () => repoPages?.pages.flatMap((page) => page.items) ?? [],
    [repoPages],
  );

  const repositories = debouncedQuery ? (searchResults ?? []) : allRepositories;

  const { mutate: createConversation, isPending } = useCreateConversation();
  const isCreatingElsewhere = useIsCreatingConversation();
  const isCreating = isPending || isCreatingElsewhere;

  React.useEffect(() => {
    if (!open) return undefined;
    const onDown = (e: MouseEvent) => {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  React.useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  const launchRepository = (repo: GitRepository) => {
    if (isCreating) return;
    createConversation(
      {
        repository: {
          name: repo.full_name,
          gitProvider: repo.git_provider,
          branch: repo.main_branch ?? "main",
        },
        entryPoint: "sidebar_cloud_menu",
      },
      {
        onSuccess: (data) => {
          setOpen(false);
          navigate(`/conversations/${data.conversation_id}`);
        },
      },
    );
  };

  const handleProviderChange = (provider: Provider) => {
    setSelectedProvider(provider);
    setLastSelectedProvider(provider);
    setQuery("");
  };

  const itemClass = dropdownMenuRowClassName;

  React.useEffect(() => {
    if (!open) return;
    if (debouncedQuery) return;
    if (!hasNextPage || isFetchingNextPage || isLoading) return;
    if (repositories.length === 0 || repositories.length >= 10) return;
    fetchNextPage();
  }, [
    open,
    debouncedQuery,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    repositories.length,
    fetchNextPage,
  ]);

  const isListLoading = debouncedQuery ? isSearchLoading : isLoading;
  const showLoadMore =
    !debouncedQuery && hasNextPage && repositories.length > 0;

  const toggleOpen = React.useCallback(() => {
    setOpen((o) => !o);
  }, []);

  const showPopover = open && (!useFixedPlacement || fixedBox !== null);

  const fixedStyle: React.CSSProperties | undefined =
    useFixedPlacement && fixedBox
      ? {
          position: "fixed",
          top: fixedBox.top,
          left: fixedBox.left,
          width: fixedBox.width,
        }
      : undefined;

  return (
    <div
      className={cn(!useFixedPlacement && "relative", className)}
      ref={popoverRef}
    >
      <span ref={triggerWrapRef} className="inline-flex">
        {trigger({
          onClick: toggleOpen,
          "aria-expanded": open,
          "aria-haspopup": "menu",
          disabled: isCreating,
        })}
      </span>

      {showPopover && (
        <div
          data-testid={popoverTestId}
          className={cn(
            NEW_CONVERSATION_DROPDOWN_SURFACE,
            !useFixedPlacement &&
              cn("absolute top-full mt-0", popoverClassName),
          )}
          style={fixedStyle}
        >
          {providers.length > 1 && (
            <div
              className="flex items-center gap-1 px-1 py-1"
              data-testid="cloud-provider-tabs"
            >
              {providers.map((provider) => {
                const isActive = provider === selectedProvider;
                return (
                  <button
                    key={provider}
                    type="button"
                    data-testid={`cloud-provider-tab-${provider}`}
                    onClick={() => handleProviderChange(provider)}
                    className={cn(
                      "flex items-center gap-1 rounded border px-2 py-1 text-xs",
                      dropdownInstantColorClassName,
                      isActive
                        ? "border-[var(--oh-border-subtle)] bg-[var(--oh-interactive-hover)] text-white"
                        : "border-transparent text-[var(--oh-text-secondary)] hover:text-white",
                    )}
                  >
                    <GitProviderIcon gitProvider={provider} />
                    <span className="capitalize">{provider}</span>
                  </button>
                );
              })}
            </div>
          )}

          <div className="px-2">
            <div className="relative">
              <SearchIcon
                width={16}
                height={16}
                aria-hidden
                className="pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 text-[var(--oh-muted)]"
              />
              <input
                type="text"
                data-testid="cloud-repo-search-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t(I18nKey.COMMON$SEARCH_REPOSITORIES)}
                disabled={!selectedProvider}
                className={cn(
                  "w-full border-0 bg-transparent py-1.5 pl-6 pr-0 text-sm text-white",
                  "outline-none placeholder:text-[var(--oh-muted)]",
                  "focus:outline-none focus:ring-0",
                  "disabled:cursor-not-allowed disabled:opacity-60",
                )}
              />
            </div>
          </div>

          <Divider inset="menu" />

          <ul
            className={cn(
              "max-h-[40vh] overflow-y-auto custom-scrollbar-always sm:max-h-[280px]",
              dropdownMenuListClassName,
            )}
          >
            {isListLoading && repositories.length === 0 && (
              <li
                className="px-2 py-2 text-sm text-[var(--oh-muted)] italic"
                data-testid="cloud-repo-loading"
              >
                {t(I18nKey.HOME$LOADING_REPOSITORIES)}
              </li>
            )}
            {isError && (
              <li
                className="px-2 py-2 text-sm text-[#F87171]"
                data-testid="cloud-repo-error"
              >
                {t(I18nKey.HOME$FAILED_TO_LOAD_REPOSITORIES)}
              </li>
            )}
            {!isListLoading &&
              !isError &&
              repositories.length === 0 &&
              !!selectedProvider && (
                <li
                  className="px-2 py-2 text-sm text-[var(--oh-muted)] italic"
                  data-testid="cloud-repo-empty"
                >
                  {t(I18nKey.GITHUB$NO_RESULTS)}
                </li>
              )}
            {repositories.map((repo) => (
              <RepoListItem
                key={`${repo.git_provider}:${repo.id}`}
                repo={repo}
                disabled={isCreating}
                onSelect={launchRepository}
                itemClass={itemClass}
              />
            ))}
            {showLoadMore && (
              <li>
                <button
                  type="button"
                  data-testid="cloud-repo-load-more"
                  disabled={isFetchingNextPage}
                  onClick={() => fetchNextPage()}
                  className={itemClass}
                >
                  <span className="text-[var(--oh-text-secondary)]">
                    {isFetchingNextPage
                      ? t(I18nKey.HOME$LOADING_MORE_REPOSITORIES)
                      : t(I18nKey.CONVERSATION$LOAD_MORE)}
                  </span>
                </button>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
