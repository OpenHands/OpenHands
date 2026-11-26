import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router";
import { I18nKey } from "#/i18n/declaration";
import type { Worklog } from "#/api/cloud/worklog-service.api";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { ErrorState } from "#/components/features/automations/error-state";
import { TabButton } from "#/components/features/conversation-panel/system-message-modal/tab-button";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { useWorklogs } from "#/hooks/query/use-worklogs";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { WorklogEntries } from "./worklog-entries";

export const WORKLOG_SEARCH_PARAM = "worklog";

/** The statuses the worklog routes use for "not licensed" and "can't be reached". */
const UNAVAILABLE_STATUSES = [403, 404, 503];

function orderWorklogs(worklogs: Worklog[]): Worklog[] {
  return [...worklogs].sort(
    (a, b) =>
      Number(b.is_default) - Number(a.is_default) ||
      a.name.localeCompare(b.name),
  );
}

export function WorklogPanel({ factoryId }: { factoryId: string }) {
  const { t } = useTranslation("openhands");
  const [searchParams, setSearchParams] = useSearchParams();
  const [showJson, setShowJson] = useState(false);
  const { data, isLoading, isError, error, refetch } = useWorklogs(factoryId);

  if (isLoading) {
    return <AutomationCardSkeleton />;
  }
  if (isError) {
    const status = getErrorStatus(error);
    if (status !== undefined && UNAVAILABLE_STATUSES.includes(status)) {
      return (
        <p data-testid="worklogs-unavailable" className="text-sm text-muted">
          {t(I18nKey.WORKLOGS$UNAVAILABLE)}
        </p>
      );
    }
    return (
      <ErrorState
        title={t(I18nKey.WORKLOGS$LOAD_ERROR)}
        onRetry={() => void refetch()}
      />
    );
  }

  const worklogs = orderWorklogs(data ?? []);
  if (worklogs.length === 0) {
    return (
      <div
        data-testid="worklogs-none"
        className={extensionModuleEmptyStateClassName}
      >
        <p className="text-sm text-contrast">{t(I18nKey.WORKLOGS$NONE)}</p>
      </div>
    );
  }

  const requestedSlug = searchParams.get(WORKLOG_SEARCH_PARAM);
  const selected =
    worklogs.find((worklog) => worklog.slug === requestedSlug) ?? worklogs[0];
  const includeJson = showJson && selected.json_entry_count > 0;

  return (
    <div className="flex flex-col gap-4">
      {worklogs.length > 1 && (
        <div
          role="tablist"
          aria-label={t(I18nKey.WORKLOGS$NAV_LABEL)}
          data-testid="worklog-tabs"
          className="flex gap-1 overflow-x-auto border-b border-border"
        >
          {worklogs.map((worklog) => (
            <TabButton
              key={worklog.slug}
              isActive={worklog.slug === selected.slug}
              className="whitespace-nowrap"
              onClick={() =>
                setSearchParams(
                  (current) => {
                    const next = new URLSearchParams(current);
                    next.set(WORKLOG_SEARCH_PARAM, worklog.slug);
                    return next;
                  },
                  { replace: true },
                )
              }
            >
              {worklog.name}
            </TabButton>
          ))}
        </div>
      )}
      {selected.json_entry_count > 0 && (
        <button
          type="button"
          data-testid="worklog-json-toggle"
          aria-pressed={includeJson}
          onClick={() => setShowJson((current) => !current)}
          className="self-start text-xs text-muted underline underline-offset-2 hover:text-contrast"
        >
          {includeJson
            ? t(I18nKey.WORKLOGS$HIDE_JSON)
            : t(I18nKey.WORKLOGS$SHOW_JSON, {
                count: selected.json_entry_count,
              })}
        </button>
      )}
      <WorklogEntries
        key={selected.slug}
        factoryId={factoryId}
        worklog={selected}
        includeJson={includeJson}
      />
    </div>
  );
}
