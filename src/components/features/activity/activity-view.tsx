import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { AlertCircle, Radio } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useActivityEventTails } from "#/hooks/query/use-activity-event-tails";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ActivityRow } from "./activity-row";
import {
  deriveSubagents,
  pickLatestActivity,
  selectActiveConversations,
} from "./activity-view-model";

const ACTIVITY_PAGE_LIMIT = 50;

function ActivityHeader() {
  const { t } = useTranslation("openhands");

  return (
    <div>
      <h1 className="text-xl font-semibold text-content">
        {t(I18nKey.ACTIVITY$TITLE)}
      </h1>
      <p className="mt-1 text-sm text-muted">{t(I18nKey.ACTIVITY$SUBTITLE)}</p>
    </div>
  );
}

function ActivityEmptyState() {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="activity-empty"
      className="flex flex-col items-center justify-center py-20 text-center"
    >
      <Radio className="size-10 text-muted" aria-hidden />
      <p className="mt-4 text-sm font-medium text-content">
        {t(I18nKey.ACTIVITY$EMPTY_TITLE)}
      </p>
      <p className="mt-1 text-sm text-muted">
        {t(I18nKey.ACTIVITY$EMPTY_BODY)}
      </p>
    </div>
  );
}

function ActivityErrorState({ onRetry }: { onRetry: () => void }) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="activity-error"
      className="flex flex-col items-center justify-center py-20 text-center"
    >
      <AlertCircle className="size-10 text-danger" aria-hidden />
      <p className="mt-4 text-sm font-medium text-content">
        {t(I18nKey.ACTIVITY$ERROR_TITLE)}
      </p>
      <BrandButton
        type="button"
        variant="secondary"
        className="mt-4"
        onClick={onRetry}
      >
        {t(I18nKey.ACTIVITY$RETRY)}
      </BrandButton>
    </div>
  );
}

function ActivityUnavailableState({ onRetry }: { onRetry: () => void }) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="activity-backend-unavailable"
      className="flex flex-col items-center justify-center py-20 text-center"
    >
      <AlertCircle className="size-10 text-warning" aria-hidden />
      <p className="mt-4 text-sm font-medium text-content">
        {t(I18nKey.ACTIVITY$UNAVAILABLE_TITLE)}
      </p>
      <p className="mt-1 max-w-md text-sm text-muted">
        {t(I18nKey.ACTIVITY$UNAVAILABLE_BODY)}
      </p>
      <BrandButton
        type="button"
        variant="secondary"
        className="mt-4"
        onClick={onRetry}
      >
        {t(I18nKey.ACTIVITY$RETRY)}
      </BrandButton>
    </div>
  );
}

/**
 * Read-only live view of every actively executing conversation. Status, cost,
 * and timestamps come from the polled conversation list; the current step and
 * the `task`-tool subagent fan-out come from each conversation's bounded event
 * tail. It performs no conversation mutations and reads only the active
 * backend's conversations.
 */
// @spec LAV-001 — Only actively executing agents are listed
// @spec LAV-005 — The view is reachable and has defined states
export function ActivityView() {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const hasBackend = !isNoBackend(active.backend);

  const { data, isLoading, isError, refetch, hasNextPage, fetchNextPage } =
    usePaginatedConversations(ACTIVITY_PAGE_LIMIT);

  const conversations = useMemo(
    () => data?.pages.flatMap((page) => page.items) ?? [],
    [data],
  );
  const activeConversations = useMemo(
    () => selectActiveConversations(conversations),
    [conversations],
  );
  const tails = useActivityEventTails(activeConversations);

  const header = <ActivityHeader />;

  if (!hasBackend) {
    return (
      <div className="min-h-full">
        <div className="mx-auto max-w-4xl p-6">
          {header}
          <ActivityUnavailableState onRetry={() => refetch()} />
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-full">
        <div className="mx-auto max-w-4xl p-6">
          {header}
          <div className="mt-6 flex flex-col gap-3">
            {Array.from({ length: 3 }).map((_, index) => (
              <div
                key={`activity-skeleton-${String(index)}`}
                className="h-24 animate-pulse rounded-lg border border-border bg-surface"
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="min-h-full">
        <div className="mx-auto max-w-4xl p-6">
          {header}
          <ActivityErrorState onRetry={() => refetch()} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-full">
      <div className="mx-auto max-w-4xl p-6">
        {header}
        <div className="mt-6 flex flex-col gap-3">
          {activeConversations.length === 0 ? (
            <ActivityEmptyState />
          ) : (
            activeConversations.map((conversation, index) => {
              const events = tails[index] ?? [];
              return (
                <ActivityRow
                  key={conversation.id}
                  conversation={conversation}
                  latestActivity={pickLatestActivity(events)}
                  subagents={deriveSubagents(events)}
                />
              );
            })
          )}
        </div>

        {hasNextPage && activeConversations.length > 0 && (
          <div className="mt-4 flex justify-center">
            <BrandButton
              type="button"
              variant="secondary"
              onClick={() => fetchNextPage()}
            >
              {t(I18nKey.ACTIVITY$LOAD_MORE)}
            </BrandButton>
          </div>
        )}
      </div>
    </div>
  );
}
