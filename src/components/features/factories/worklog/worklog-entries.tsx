import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Worklog } from "#/api/cloud/worklog-service.api";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { ErrorState } from "#/components/features/automations/error-state";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { useWorklogEntries } from "#/hooks/query/use-worklogs";
import { WorklogEntryCard } from "./worklog-entry-card";

interface WorklogEntriesProps {
  factoryId: string;
  worklog: Worklog;
  includeJson: boolean;
}

/** One worklog's entries, newest first, with an explicit "load older" page. */
export function WorklogEntries({
  factoryId,
  worklog,
  includeJson,
}: WorklogEntriesProps) {
  const { t } = useTranslation("openhands");
  const {
    data,
    isLoading,
    isError,
    refetch,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
  } = useWorklogEntries(factoryId, worklog, { includeJson });

  if (isLoading) {
    return (
      <div className="flex flex-col gap-3">
        <AutomationCardSkeleton />
        <AutomationCardSkeleton />
      </div>
    );
  }
  if (isError || !data) {
    return (
      <ErrorState
        title={t(I18nKey.WORKLOGS$LOAD_ERROR)}
        onRetry={() => void refetch()}
      />
    );
  }

  const entries = data.pages.flatMap((page) => page.items);
  if (entries.length === 0) {
    return (
      <div
        data-testid="worklog-empty"
        className={extensionModuleEmptyStateClassName}
      >
        <p className="text-sm text-contrast">{t(I18nKey.WORKLOGS$EMPTY)}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <ul
        aria-label={t(I18nKey.WORKLOGS$ENTRIES_LABEL)}
        className="flex flex-col gap-3"
      >
        {entries.map((entry) => (
          <WorklogEntryCard key={entry.id} entry={entry} />
        ))}
      </ul>
      {hasNextPage && (
        <button
          type="button"
          data-testid="worklog-load-more"
          disabled={isFetchingNextPage}
          onClick={() => void fetchNextPage()}
          className="self-center rounded-lg border border-border px-4 py-2 text-sm text-contrast hover:bg-surface-raised disabled:opacity-50"
        >
          {t(I18nKey.WORKLOGS$LOAD_MORE)}
        </button>
      )}
    </div>
  );
}
