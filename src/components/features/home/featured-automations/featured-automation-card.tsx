import { useTranslation } from "react-i18next";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { NavigationLink } from "#/components/shared/navigation-link";
import type { LatestAutomationRunState } from "#/hooks/query/use-latest-automation-runs";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { I18nKey } from "#/i18n/declaration";
import ClockIcon from "#/icons/clock.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import { AutomationRunStatus, type Automation } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
import {
  getLastRunTimestamp,
  getTriggerSummary,
} from "./automation-run-health";

interface FeaturedAutomationCardProps {
  automation: Automation;
  runState: LatestAutomationRunState;
}

export function FeaturedAutomationCard({
  automation,
  runState,
}: FeaturedAutomationCardProps) {
  const { t, i18n } = useTranslation("openhands");
  const { latestRun, isLoading, isError } = runState;
  const conversationId = latestRun?.conversation_id ?? null;
  // The conversation title is the closest live "result" line the platform
  // offers (the automation API has no run summary field). Degrades to a
  // generic link label when the conversation is missing or has no title.
  const { data: conversation } = useUserConversation(conversationId);

  const timestamp = latestRun ? getLastRunTimestamp(latestRun) : null;
  const isTerminal =
    latestRun?.status === AutomationRunStatus.COMPLETED ||
    latestRun?.status === AutomationRunStatus.FAILED;
  const TriggerIcon =
    automation.trigger.type === "event" ? GlobeIcon : ClockIcon;

  return (
    <article className="rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-4">
      <div className="flex items-start justify-between gap-3">
        <NavigationLink
          to={`/automations/${automation.id}`}
          title={t(I18nKey.FEATURED_AUTOMATIONS$VIEW_DETAILS)}
          className="min-w-0 font-medium text-[var(--oh-foreground)] hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
        >
          <span className="block truncate">{automation.name}</span>
        </NavigationLink>
        {latestRun ? <RunStatusBadge status={latestRun.status} /> : null}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[var(--oh-text-secondary)]">
        <span className="flex items-center gap-1">
          <TriggerIcon className="size-3 shrink-0" aria-hidden="true" />
          {getTriggerSummary(automation)}
        </span>
        {timestamp ? (
          <span>
            {t(I18nKey.AUTOMATIONS$DETAIL$LAST_RUN)}:{" "}
            {formatRelativeTime(timestamp, i18n.language, t)}
          </span>
        ) : null}
      </div>

      <div className="mt-3 space-y-2 rounded-md border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)] p-3 text-sm">
        {isLoading ? (
          <div className="space-y-2" aria-hidden="true">
            <div className="h-4 w-3/4 animate-pulse rounded bg-surface-raised motion-reduce:animate-none" />
            <div className="h-4 w-1/2 animate-pulse rounded bg-surface-raised motion-reduce:animate-none" />
          </div>
        ) : null}

        {!isLoading && isError ? (
          <p className="text-[var(--oh-text-secondary)]">
            {t(I18nKey.FEATURED_AUTOMATIONS$STATUS_UNAVAILABLE)}
          </p>
        ) : null}

        {!isLoading && !isError && !latestRun ? (
          <p className="text-[var(--oh-text-secondary)]">
            {t(I18nKey.AUTOMATIONS$DETAIL$NO_RUNS)}
          </p>
        ) : null}

        {latestRun ? (
          <>
            {latestRun.status === AutomationRunStatus.FAILED &&
            latestRun.error_detail ? (
              <p className="line-clamp-3 text-[var(--oh-status-error)]">
                {latestRun.error_detail}
              </p>
            ) : null}

            {conversationId ? (
              <NavigationLink
                to={`/conversations/${conversationId}`}
                className="block text-[var(--oh-foreground)] underline underline-offset-4 hover:text-[var(--oh-text-secondary)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
              >
                <span className="block truncate">
                  {conversation?.title ||
                    t(I18nKey.FEATURED_AUTOMATIONS$VIEW_CONVERSATION)}
                </span>
              </NavigationLink>
            ) : null}

            {!conversationId && isTerminal ? (
              <p className="text-xs italic text-[var(--oh-text-secondary)]">
                ({t(I18nKey.AUTOMATIONS$DETAIL$NO_CONVERSATION)})
              </p>
            ) : null}
          </>
        ) : null}
      </div>
    </article>
  );
}
