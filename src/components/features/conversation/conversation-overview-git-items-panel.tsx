import React from "react";
import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";
import { GitProviderItemsService } from "#/api/git-provider-items-service";
import {
  useRepositoryIssues,
  useRepositoryPullRequests,
} from "#/hooks/query/use-repository-git-items";
import { useConversationPrimaryRepository } from "#/hooks/use-conversation-primary-repository";
import { I18nKey } from "#/i18n/declaration";
import { cn, getProviderName } from "#/utils/utils";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { CONVERSATION_OVERVIEW_DRAWER_SECTION } from "./conversation-overview-drawer.types";

interface ConversationOverviewGitItemsPanelProps {
  kind: "pull_requests" | "issues";
}

const EXTERNAL_LINK_ICON_CLASSNAME = cn(
  "size-3.5 shrink-0 text-[var(--oh-muted)]",
  "opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100",
);

function useGitItemsExternalListUrl(kind: "pull_requests" | "issues") {
  const { repository, provider } = useConversationPrimaryRepository();

  if (!repository || !provider) {
    return null;
  }

  return kind === CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests
    ? GitProviderItemsService.constructPullRequestsListUrl(provider, repository)
    : GitProviderItemsService.constructIssuesListUrl(provider, repository);
}

export function ConversationOverviewGitItemsHeaderLink({
  kind,
}: ConversationOverviewGitItemsPanelProps) {
  const { t } = useTranslation("openhands");
  const { provider } = useConversationPrimaryRepository();
  const externalListUrl = useGitItemsExternalListUrl(kind);

  if (!externalListUrl || !provider) {
    return null;
  }

  return (
    <a
      href={externalListUrl}
      target="_blank"
      rel="noreferrer"
      data-testid={`conversation-overview-${kind}-open-external`}
      className={cn(
        "inline-flex h-7 min-h-7 shrink-0 items-center gap-1.5 whitespace-nowrap",
        "px-1 text-xs text-[var(--oh-muted)] transition-colors",
        "hover:text-[var(--oh-foreground)]",
      )}
    >
      {t(I18nKey.CONVERSATION$OVERVIEW_VIEW_ON_PROVIDER, {
        provider: getProviderName(provider),
      })}
      <ExternalLink className="size-3.5" aria-hidden />
    </a>
  );
}

export function ConversationOverviewGitItemsPanel({
  kind,
}: ConversationOverviewGitItemsPanelProps) {
  const { t } = useTranslation("openhands");
  const { repository, provider, isConnected } =
    useConversationPrimaryRepository();

  const pullRequestsQuery = useRepositoryPullRequests(repository, provider);
  const issuesQuery = useRepositoryIssues(repository, provider);

  const query =
    kind === CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests
      ? pullRequestsQuery
      : issuesQuery;

  if (!isConnected || !repository || !provider) {
    return (
      <p
        data-testid="conversation-overview-git-items-unavailable"
        className={cn(
          extensionModuleEmptyStateClassName,
          "px-4 py-6 text-center text-sm",
        )}
      >
        {t(I18nKey.CONVERSATION$OVERVIEW_GIT_UNAVAILABLE)}
      </p>
    );
  }

  return (
    <div
      data-testid={`conversation-overview-${kind}-panel`}
      className="min-h-0 flex-1 overflow-y-auto px-2 py-2"
    >
      {query.isLoading ? (
        <p className="px-2 py-4 text-sm text-[var(--oh-muted)]">
          {t(I18nKey.HOME$LOADING)}
        </p>
      ) : query.isError ? (
        <p
          data-testid={`conversation-overview-${kind}-error`}
          className={cn(
            extensionModuleEmptyStateClassName,
            "px-2 py-6 text-center text-sm",
          )}
        >
          {t(I18nKey.CONVERSATION$OVERVIEW_GIT_ITEMS_ERROR)}
        </p>
      ) : query.data && query.data.length > 0 ? (
        <ul className="flex flex-col gap-0.5">
          {query.data.map((item) => (
            <li key={item.id}>
              <a
                href={item.url}
                target="_blank"
                rel="noreferrer"
                data-testid={`conversation-overview-${kind}-item-${item.number}`}
                className={cn(
                  "group flex min-w-0 items-center gap-2 rounded-md px-2 py-2",
                  "transition-colors hover:bg-white/5",
                )}
              >
                <span className="shrink-0 text-sm tabular-nums text-[var(--oh-muted)]">
                  #{item.number}
                </span>
                <span className="min-w-0 flex-1 truncate text-sm text-[var(--oh-foreground)]">
                  {item.title}
                </span>
                <ExternalLink
                  className={EXTERNAL_LINK_ICON_CLASSNAME}
                  aria-hidden
                />
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p
          data-testid={`conversation-overview-${kind}-empty`}
          className={cn(
            extensionModuleEmptyStateClassName,
            "px-2 py-6 text-center text-sm",
          )}
        >
          {kind === CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests
            ? t(I18nKey.CONVERSATION$OVERVIEW_PULL_REQUESTS_EMPTY)
            : t(I18nKey.CONVERSATION$OVERVIEW_ISSUES_EMPTY)}
        </p>
      )}
    </div>
  );
}
