import { Pin } from "lucide-react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { NavigationLink } from "#/components/shared/navigation-link";
import {
  UNKNOWN_RUN_STATE,
  useHomeAutomations,
} from "#/hooks/query/use-home-automations";
import { useHomePinnedAutomations } from "#/hooks/use-home-pinned-automations";
import { I18nKey } from "#/i18n/declaration";
import {
  buildHomeAutomationActivityItems,
  hrefForActivityItem,
} from "./home-automation-activity";

/**
 * Dashboard grid of pinned automations rendered above Recent Automation
 * Activity on the home page. Empty when nothing pinned resolves against
 * currently enabled live automations.
 */
export function PinnedAutomationsDashboard() {
  const { t, i18n } = useTranslation("openhands");
  const { pinnedIds, unpin } = useHomePinnedAutomations();
  const {
    isBackendHealthy,
    isHealthLoading,
    isError,
    enabledAutomations,
    runStates,
  } = useHomeAutomations();

  const itemsById = useMemo(() => {
    const items = buildHomeAutomationActivityItems(
      enabledAutomations,
      runStates,
      i18n.language,
      t,
      UNKNOWN_RUN_STATE,
    );
    return new Map(items.map((item) => [item.id, item]));
  }, [enabledAutomations, runStates, i18n.language, t]);

  // Stored ids that no longer match an enabled automation (deleted, disabled,
  // or from another backend/org) are kept in storage but not rendered.
  const pinnedItems = pinnedIds
    .map((id) => itemsById.get(id))
    .filter((item): item is NonNullable<typeof item> => Boolean(item));

  if (
    isHealthLoading ||
    !isBackendHealthy ||
    isError ||
    pinnedItems.length === 0
  ) {
    return null;
  }

  return (
    <section
      data-testid="pinned-automations-dashboard"
      aria-label={t(I18nKey.FEATURED_AUTOMATIONS$PINNED_DASHBOARD_LABEL)}
      className="w-full"
    >
      <h2 className="mb-2 text-sm font-medium text-[var(--oh-foreground)]">
        {t(I18nKey.FEATURED_AUTOMATIONS$PINNED_TITLE)}
      </h2>

      <div role="list" className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {pinnedItems.map((item) => {
          const metaLine = [item.triggerSummary, item.whenLabel]
            .filter(Boolean)
            .join(" · ");

          return (
            <article
              key={item.id}
              role="listitem"
              data-testid={`pinned-automation-card-${item.id}`}
              className="relative flex flex-col gap-2 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-3"
            >
              <div className="flex items-start justify-between gap-2">
                <NavigationLink
                  to={hrefForActivityItem(item)}
                  className="min-w-0 flex-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
                >
                  <span className="block truncate text-sm font-medium text-[var(--oh-foreground)]">
                    {item.name}
                  </span>
                  {metaLine ? (
                    <span className="mt-0.5 block truncate text-xs text-[var(--oh-text-secondary)]">
                      {metaLine}
                    </span>
                  ) : null}
                </NavigationLink>

                <button
                  type="button"
                  data-testid={`unpin-automation-${item.id}`}
                  aria-label={t(I18nKey.FEATURED_AUTOMATIONS$UNPIN)}
                  className="inline-flex size-7 shrink-0 items-center justify-center rounded-md text-[var(--oh-muted)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]"
                  onClick={() => unpin(item.id)}
                >
                  <Pin className="size-3.5 fill-current" aria-hidden="true" />
                </button>
              </div>

              {item.status ? (
                <div className="mt-auto">
                  <RunStatusBadge status={item.status} />
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
