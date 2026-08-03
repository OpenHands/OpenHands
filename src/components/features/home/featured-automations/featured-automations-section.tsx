import { useState } from "react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useAutomations } from "#/hooks/query/use-automations";
import {
  useLatestAutomationRuns,
  type LatestAutomationRunState,
} from "#/hooks/query/use-latest-automation-runs";
import { I18nKey } from "#/i18n/declaration";
import PlusIcon from "#/icons/plus.svg?react";
import SparkleIcon from "#/icons/sparkle.svg?react";
import type { Automation } from "#/types/automation";
import { FeaturedAutomationCard } from "./featured-automation-card";
import { FeaturedAutomationChip } from "./featured-automation-chip";

/** Bounds the per-automation latest-run request fan-out on the home page. */
const MAX_AUTOMATION_CHIPS = 20;

const UNKNOWN_RUN_STATE: LatestAutomationRunState = {
  latestRun: null,
  isLoading: true,
  isError: false,
};

export const HOME_FEATURED_AUTOMATION_IDS_KEY =
  "oh:home-featured-automation-ids";

function getStoredFeaturedAutomationIds(): string[] {
  if (typeof window === "undefined") return [];

  try {
    const raw = window.sessionStorage.getItem(HOME_FEATURED_AUTOMATION_IDS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((id): id is string => typeof id === "string");
  } catch {
    return [];
  }
}

function setStoredFeaturedAutomationIds(ids: string[]): void {
  if (typeof window === "undefined") return;

  try {
    if (ids.length > 0) {
      window.sessionStorage.setItem(
        HOME_FEATURED_AUTOMATION_IDS_KEY,
        JSON.stringify(ids),
      );
    } else {
      window.sessionStorage.removeItem(HOME_FEATURED_AUTOMATION_IDS_KEY);
    }
  } catch {
    // sessionStorage may be unavailable in private browsing contexts.
  }
}

/**
 * Home-page landing dashboard for automations: one health-indicated chip per
 * enabled automation, and an expanded card for each automation the user has
 * featured. Self-gating — renders nothing while loading, on errors, when the
 * automation service is unavailable (e.g. a local deployment without the
 * automation sidecar), or when there are no enabled automations.
 */
export function FeaturedAutomationsSection() {
  const { t } = useTranslation("openhands");
  const { data: healthData } = useAutomationHealth();
  const isBackendHealthy = healthData?.status === "ok";
  const { data: automationsData, isError } = useAutomations({
    limit: 50,
    offset: 0,
    enabled: isBackendHealthy,
  });
  // Featured selection is deliberately per-tab ("persistent in session"):
  // it survives reloads and in-app navigation but not a new tab.
  const [featuredIds, setFeaturedIds] = useState<string[]>(
    getStoredFeaturedAutomationIds,
  );

  const enabledAutomations = (automationsData?.automations ?? [])
    .filter((automation) => automation.enabled)
    .slice(0, MAX_AUTOMATION_CHIPS);

  const runStates = useLatestAutomationRuns(enabledAutomations);

  if (!isBackendHealthy || isError || enabledAutomations.length === 0) {
    return null;
  }

  const toggleFeatured = (automationId: string) => {
    const next = featuredIds.includes(automationId)
      ? featuredIds.filter((id) => id !== automationId)
      : [...featuredIds, automationId];
    setFeaturedIds(next);
    setStoredFeaturedAutomationIds(next);
  };

  // Stored ids that no longer match a listed automation (deleted, disabled,
  // or from another backend/org) are kept in storage but not rendered.
  const enabledById = new Map(
    enabledAutomations.map((automation) => [automation.id, automation]),
  );
  const featuredAutomations = featuredIds
    .map((id) => enabledById.get(id))
    .filter((automation): automation is Automation => Boolean(automation));

  return (
    <section
      aria-labelledby="featured-automations-heading"
      data-testid="featured-automations-section"
      className="mx-auto w-full max-w-5xl rounded-xl border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)] p-4"
    >
      <h2
        id="featured-automations-heading"
        className="mb-3 text-sm font-medium text-[var(--oh-foreground)]"
      >
        {t(I18nKey.FEATURED_AUTOMATIONS$SECTION_TITLE)}
      </h2>

      <div
        role="group"
        aria-label={t(I18nKey.FEATURED_AUTOMATIONS$CHIP_GROUP_LABEL)}
        className="flex flex-wrap gap-2"
      >
        {enabledAutomations.map((automation) => (
          <FeaturedAutomationChip
            key={automation.id}
            automation={automation}
            runState={runStates.get(automation.id) ?? UNKNOWN_RUN_STATE}
            isFeatured={featuredIds.includes(automation.id)}
            onToggle={toggleFeatured}
          />
        ))}
        <StyledTooltip
          content={t(I18nKey.FEATURED_AUTOMATIONS$MANAGE)}
          placement="bottom"
        >
          <NavigationLink
            to="/automations"
            aria-label={t(I18nKey.FEATURED_AUTOMATIONS$MANAGE)}
            className="inline-flex items-center justify-center rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-3 py-2 text-[var(--oh-foreground)] transition-colors hover:bg-[var(--oh-interactive-hover)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
          >
            <PlusIcon className="size-4" aria-hidden="true" />
          </NavigationLink>
        </StyledTooltip>
      </div>

      {featuredAutomations.length > 0 ? (
        <div className="mt-4 border-t border-[var(--oh-border-subtle)] pt-4">
          <div className="mb-3 flex items-center gap-2">
            <SparkleIcon
              className="size-4 text-[var(--oh-status-success)]"
              aria-hidden="true"
            />
            <h3 className="text-sm font-medium text-[var(--oh-foreground)]">
              {t(I18nKey.FEATURED_AUTOMATIONS$FEATURED_HEADING)}
            </h3>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {featuredAutomations.map((automation) => (
              <FeaturedAutomationCard
                key={automation.id}
                automation={automation}
                runState={runStates.get(automation.id) ?? UNKNOWN_RUN_STATE}
              />
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
