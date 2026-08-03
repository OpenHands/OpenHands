import { Pin } from "lucide-react";
import { useTranslation } from "react-i18next";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useHomePinnedAutomations } from "#/hooks/use-home-pinned-automations";
import { I18nKey } from "#/i18n/declaration";
import type { HomeAutomationActivityExample } from "./home-automation-activity-examples";

function hrefForExample(example: HomeAutomationActivityExample): string {
  return example.conversationId
    ? `/conversations/${example.conversationId}`
    : `/automations/${example.id}`;
}

/**
 * Dashboard grid of pinned automations rendered above Recent Automation
 * Activity on the home page. Empty when nothing is pinned.
 */
export function PinnedAutomationsDashboard() {
  const { t } = useTranslation("openhands");
  const { pinnedExamples, unpin } = useHomePinnedAutomations();

  if (pinnedExamples.length === 0) {
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
        {pinnedExamples.map((example) => (
          <article
            key={example.id}
            role="listitem"
            data-testid={`pinned-automation-card-${example.id}`}
            className="relative flex flex-col gap-2 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <NavigationLink
                to={hrefForExample(example)}
                className="min-w-0 flex-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
              >
                <span className="block truncate text-sm font-medium text-[var(--oh-foreground)]">
                  {example.name}
                </span>
                <span className="mt-0.5 block truncate text-xs text-[var(--oh-text-secondary)]">
                  {example.triggerSummary}
                  {}
                  {" · "}
                  {example.whenLabel}
                </span>
              </NavigationLink>

              <button
                type="button"
                data-testid={`unpin-automation-${example.id}`}
                aria-label={t(I18nKey.FEATURED_AUTOMATIONS$UNPIN)}
                className="inline-flex size-7 shrink-0 items-center justify-center rounded-md text-[var(--oh-muted)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]"
                onClick={() => unpin(example.id)}
              >
                <Pin className="size-3.5 fill-current" aria-hidden="true" />
              </button>
            </div>

            <div className="mt-auto">
              <RunStatusBadge status={example.status} />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
