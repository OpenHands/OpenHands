import { CircleX, Clock } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { ConversationPullRequest } from "#/api/git-provider-pull-request-status-service";
import {
  PULL_REQUEST_CHECKS_LABEL,
  PULL_REQUEST_LIFECYCLE_LABEL,
  PULL_REQUEST_UNAVAILABLE_LABEL,
} from "#/components/features/conversation/pull-request-status-labels";
import { useConversationPullRequests } from "#/hooks/query/use-conversation-pull-requests";
import { I18nKey } from "#/i18n/declaration";
import PrIcon from "#/icons/u-pr.svg?react";
import type { Provider } from "#/types/settings";
import { cn, getProviderName } from "#/utils/utils";
import {
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME,
} from "./conversation-card-meta-chip";

interface ConversationPullRequestIndicatorProps {
  repository: string | null;
  provider: Provider | null;
  prNumbers?: readonly number[] | null;
}

/**
 * Code-delivery badge for a conversation row, kept apart from the execution
 * status dot: a finished conversation can still have an open PR with failing
 * checks. Only PRs the conversation tracks are shown, so a row never claims
 * a PR it did not record, and rows without one make no provider requests.
 */
export function ConversationPullRequestIndicator({
  repository,
  provider,
  prNumbers,
}: ConversationPullRequestIndicatorProps) {
  const { t } = useTranslation("openhands");
  const hasTrackedPullRequests = Boolean(prNumbers?.length);
  const query = useConversationPullRequests({
    repository: hasTrackedPullRequests ? repository : null,
    provider: hasTrackedPullRequests ? provider : null,
    branch: null,
    prNumbers,
  });

  if (!hasTrackedPullRequests || !repository || !provider || query.isPending) {
    return null;
  }

  const describe = (pullRequest: ConversationPullRequest) =>
    pullRequest.unavailableReason
      ? t(I18nKey.CONVERSATION$PR_INDICATOR_UNREADABLE, {
          number: pullRequest.number,
          repository: pullRequest.repository,
          reason: t(
            PULL_REQUEST_UNAVAILABLE_LABEL[pullRequest.unavailableReason],
          ),
        })
      : t(I18nKey.CONVERSATION$PR_INDICATOR_DESCRIPTION, {
          number: pullRequest.number,
          repository: pullRequest.repository,
          lifecycle: t(PULL_REQUEST_LIFECYCLE_LABEL[pullRequest.lifecycle]),
          checks: t(PULL_REQUEST_CHECKS_LABEL[pullRequest.checks]),
        });

  const result = query.data;
  let label = t(I18nKey.CONVERSATION$PR_INDICATOR_UNKNOWN);
  let description: string;
  let readable: ConversationPullRequest[] = [];
  if (!result) {
    description = t(PULL_REQUEST_UNAVAILABLE_LABEL.error);
  } else if (result.status === "unsupported_provider") {
    description = t(I18nKey.CONVERSATION$PR_STATUS_UNSUPPORTED, {
      provider: getProviderName(provider),
    });
  } else if (result.status === "unavailable") {
    description = t(PULL_REQUEST_UNAVAILABLE_LABEL[result.reason]);
  } else {
    const { pullRequests, omittedCount } = result;
    readable = pullRequests.filter((pr) => !pr.unavailableReason);
    const total = pullRequests.length + omittedCount;
    if (total > 1) {
      label = t(I18nKey.CONVERSATION$PR_INDICATOR_COUNT, { count: total });
    } else if (readable.length === 1) {
      label = t(PULL_REQUEST_LIFECYCLE_LABEL[readable[0].lifecycle]);
    }
    const lines = pullRequests.map(describe);
    if (omittedCount > 0) {
      lines.push(t(I18nKey.CONVERSATION$PR_OMITTED, { count: omittedCount }));
    }
    description = lines.join("\n");
  }

  // A failure anywhere outranks pending elsewhere, as in the PR's own checks.
  const checksMarker = readable.some((pr) => pr.checks === "failing")
    ? "failing"
    : readable.some((pr) => pr.checks === "pending")
      ? "pending"
      : null;

  return (
    <span
      data-testid="conversation-card-pull-requests"
      title={description}
      className={cn(
        CONVERSATION_CARD_META_CHIP_CLASSNAME,
        checksMarker === "failing" && "text-danger",
      )}
    >
      <span
        className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
        aria-hidden
      >
        <PrIcon className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME} />
      </span>
      <span className="truncate leading-4" aria-hidden>
        {label}
      </span>
      {checksMarker ? (
        <span
          data-testid={`conversation-card-pull-requests-checks-${checksMarker}`}
          className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
          aria-hidden
        >
          {checksMarker === "failing" ? (
            <CircleX className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME} />
          ) : (
            <Clock className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME} />
          )}
        </span>
      ) : null}
      <span className="sr-only">{description}</span>
    </span>
  );
}
