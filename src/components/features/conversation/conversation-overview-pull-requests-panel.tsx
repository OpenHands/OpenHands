import type { ReactNode } from "react";
import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { ConversationPullRequest } from "#/api/git-provider-pull-request-status-service";
import { useActiveConversationPullRequests } from "#/hooks/use-active-conversation-pull-requests";
import { I18nKey } from "#/i18n/declaration";
import type { Provider } from "#/types/settings";
import { cn, getProviderName } from "#/utils/utils";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import {
  PULL_REQUEST_CHECKS_LABEL,
  PULL_REQUEST_LIFECYCLE_LABEL,
  PULL_REQUEST_MERGE_LABEL,
  PULL_REQUEST_REVIEW_LABEL,
  PULL_REQUEST_UNAVAILABLE_LABEL,
} from "./pull-request-status-labels";
import { CONVERSATION_CARD_META_CHIP_CLASSNAME } from "#/components/features/conversation-panel/conversation-card/conversation-card-meta-chip";

const EXTERNAL_LINK_ICON_CLASSNAME = cn(
  "size-3.5 shrink-0 text-muted",
  "opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100",
);
const MESSAGE_CLASSNAME = cn(
  extensionModuleEmptyStateClassName,
  "px-2 py-6 text-center text-sm",
);
const NOTE_CLASSNAME = "px-2 py-1 text-xs text-muted";

type Tone = "neutral" | "attention";

const TONE_CLASSNAME: Record<Tone, string> = {
  neutral: "text-muted",
  attention: "text-danger",
};

function StatusChip({ label, tone }: { label: I18nKey; tone: Tone }) {
  const { t } = useTranslation("openhands");
  return (
    <span
      className={cn(
        CONVERSATION_CARD_META_CHIP_CLASSNAME,
        TONE_CLASSNAME[tone],
      )}
    >
      {t(label)}
    </span>
  );
}

function PullRequestRow({
  pullRequest,
}: {
  pullRequest: ConversationPullRequest;
}) {
  const { t } = useTranslation("openhands");
  const { number, url, title, unavailableReason, mergeable } = pullRequest;

  return (
    <li data-testid={`conversation-pull-request-${number}`}>
      <a
        href={url}
        target="_blank"
        rel="noreferrer"
        className={cn(
          "group flex min-w-0 items-center gap-2 rounded-md px-2 pt-2",
          "transition-colors hover:bg-contrast/5",
        )}
      >
        <span className="shrink-0 text-sm tabular-nums text-muted">
          #{number}
        </span>
        <span className="min-w-0 flex-1 truncate text-sm text-foreground">
          {title ??
            (unavailableReason
              ? t(PULL_REQUEST_UNAVAILABLE_LABEL[unavailableReason])
              : null)}
        </span>
        <ExternalLink className={EXTERNAL_LINK_ICON_CLASSNAME} aria-hidden />
      </a>
      <div className="flex min-w-0 flex-wrap items-center gap-1 px-2 pb-2 pt-1">
        <span className="min-w-0 truncate text-xs text-muted">
          {pullRequest.repository}
        </span>
        {unavailableReason ? null : (
          <>
            <StatusChip
              label={PULL_REQUEST_LIFECYCLE_LABEL[pullRequest.lifecycle]}
              tone="neutral"
            />
            <StatusChip
              label={PULL_REQUEST_CHECKS_LABEL[pullRequest.checks]}
              tone={pullRequest.checks === "failing" ? "attention" : "neutral"}
            />
            <StatusChip
              label={PULL_REQUEST_REVIEW_LABEL[pullRequest.review]}
              tone={
                pullRequest.review === "changes_requested"
                  ? "attention"
                  : "neutral"
              }
            />
            {mergeable === "not_applicable" ? null : (
              <StatusChip
                label={PULL_REQUEST_MERGE_LABEL[mergeable]}
                tone={mergeable === "conflicting" ? "attention" : "neutral"}
              />
            )}
          </>
        )}
      </div>
    </li>
  );
}

/**
 * The open conversation's own PRs with their current provider state. Every
 * outcome other than a loaded list says what happened, so an empty panel
 * never stands in for "unknown".
 */
export function ConversationOverviewPullRequestsPanel({
  provider,
}: {
  provider: Provider;
}) {
  const { t } = useTranslation("openhands");
  const query = useActiveConversationPullRequests();
  const result = query.data;

  let content: ReactNode;
  if (query.isLoading) {
    content = (
      <p className="px-2 py-4 text-sm text-muted">{t(I18nKey.HOME$LOADING)}</p>
    );
  } else if (query.isError || !result) {
    content = (
      <p className={MESSAGE_CLASSNAME}>
        {t(I18nKey.CONVERSATION$OVERVIEW_GIT_ITEMS_ERROR)}
      </p>
    );
  } else if (result.status === "unsupported_provider") {
    content = (
      <p className={MESSAGE_CLASSNAME}>
        {t(I18nKey.CONVERSATION$PR_STATUS_UNSUPPORTED, {
          provider: getProviderName(provider),
        })}
      </p>
    );
  } else if (result.status === "unavailable") {
    content = (
      <p className={MESSAGE_CLASSNAME}>
        {t(PULL_REQUEST_UNAVAILABLE_LABEL[result.reason])}
      </p>
    );
  } else if (result.pullRequests.length === 0) {
    content = (
      <p className={MESSAGE_CLASSNAME}>
        {t(I18nKey.CONVERSATION$PR_NONE_FOR_CONVERSATION)}
      </p>
    );
  } else {
    content = (
      <>
        {result.source === "branch" ? (
          <p className={NOTE_CLASSNAME}>
            {t(I18nKey.CONVERSATION$PR_FROM_BRANCH)}
          </p>
        ) : null}
        <ul className="flex flex-col gap-0.5">
          {result.pullRequests.map((pullRequest) => (
            <PullRequestRow
              key={`${pullRequest.repository}#${pullRequest.number}`}
              pullRequest={pullRequest}
            />
          ))}
        </ul>
        {result.omittedCount > 0 ? (
          <p className={NOTE_CLASSNAME}>
            {t(I18nKey.CONVERSATION$PR_OMITTED, {
              count: result.omittedCount,
            })}
          </p>
        ) : null}
      </>
    );
  }

  return (
    <div
      data-testid="conversation-overview-pull_requests-panel"
      className="min-h-0 flex-1 overflow-y-auto px-2 py-2"
    >
      {content}
    </div>
  );
}
