import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { KebabMenu } from "./kebab-menu";
import { useHasPermission } from "#/hooks/use-has-permission";
import { useNavigation } from "#/context/navigation-context";
import PlayIcon from "#/icons/play.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import { Zap } from "lucide-react";
import { SkillCardPillRow } from "#/components/features/skills/skill-card-pill-row";
import { cn } from "#/utils/utils";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { buildAutomationMetadataPills } from "./build-automation-pills";
import { buildAutomationMenuItems } from "./build-automation-menu-items";
import { automationRunNowTextButtonClassName } from "./automation-action-button-classes";
import { AutomationHealthBadge } from "./automation-health-badge";
import { AutomationRunStats, lastRunText } from "./automation-run-insights";
import {
  deriveAutomationHealth,
  type RunSummaryState,
} from "#/manifests/automation-insights";
import type { InterfaceListInsights } from "#/manifests/types";

/** Run insights shown when the manifest declares the dashboard surface. */
export interface AutomationInsightsProps {
  spec: InterfaceListInsights;
  state: RunSummaryState | undefined;
}

interface AutomationCardProps {
  automation: Automation;
  onToggle: (id: string, enabled: boolean) => void;
  onRunNow: (id: string) => void;
  isRunPending?: boolean;
  onDelete: (id: string) => void;
  onExport: (automation: Automation) => void;
  onEdit?: (id: string) => void;
  insights?: AutomationInsightsProps;
}

export function AutomationCard({
  automation,
  onToggle,
  onRunNow,
  isRunPending = false,
  onDelete,
  onExport,
  onEdit,
  insights,
}: AutomationCardProps) {
  const { navigate } = useNavigation();
  const { t } = useTranslation("openhands");
  const canManage = useHasPermission("manage_automations");

  const scheduleLabel =
    automation.trigger.schedule_human || automation.trigger.type;
  const pills = useMemo(
    () => buildAutomationMetadataPills(automation, scheduleLabel),
    [automation, scheduleLabel],
  );

  const handleView = () => {
    navigate?.(`/automations/${automation.id}`);
  };

  const menuItems = buildAutomationMenuItems({
    automation,
    t,
    canManage,
    onRunNow,
    isRunPending,
    onView: handleView,
    onExport,
    onEdit,
    onToggle,
    onDelete,
  });

  const handleCardClick = () => {
    handleView();
  };

  return (
    <div
      role="link"
      tabIndex={0}
      data-testid={`automation-card-${automation.id}`}
      onClick={handleCardClick}
      onKeyDown={(e) => {
        if (e.key === "Enter") handleCardClick();
      }}
      className={cn(
        "flex min-w-0 flex-col gap-3 overflow-hidden p-4 text-left",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
      )}
    >
      <header className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <h3 className="flex items-center gap-2 truncate text-sm font-semibold text-white">
            {automation.trigger.type === "event" ? (
              <Zap className="size-4 shrink-0 text-muted" aria-hidden="true" />
            ) : (
              <ClockIcon className="size-4 shrink-0 text-muted" />
            )}
            <span className="truncate">{automation.name}</span>
          </h3>
          {automation.prompt ? (
            <p className="line-clamp-2 text-xs leading-relaxed text-tertiary-light">
              {automation.prompt}
            </p>
          ) : null}
        </div>

        <div className="flex shrink-0 items-center gap-0.5">
          {canManage ? (
            <button
              type="button"
              data-testid={`automation-run-now-${automation.id}`}
              aria-busy={isRunPending}
              disabled={isRunPending || !automation.enabled}
              onClick={(event) => {
                event.stopPropagation();
                onRunNow(automation.id);
              }}
              className={automationRunNowTextButtonClassName}
            >
              <PlayIcon className="size-3.5 shrink-0" aria-hidden />
              {t(I18nKey.AUTOMATIONS$RUN_NOW)}
            </button>
          ) : null}
          <KebabMenu items={menuItems} />
        </div>
      </header>

      {insights ? (
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          <AutomationHealthBadge
            health={deriveAutomationHealth(automation, insights.state)}
            labels={insights.spec.health}
          />
          <span data-testid={`automation-last-run-${automation.id}`}>
            {`${insights.spec.lastRun.label} ${lastRunText(
              insights.state?.summary?.latestRun?.started_at ??
                automation.last_triggered_at,
              insights.spec.lastRun,
              t(I18nKey.CONVERSATION$AGO),
            )}`}
          </span>
        </div>
      ) : null}

      {pills.length > 0 ? (
        <SkillCardPillRow
          pills={pills}
          testId={`automation-pills-${automation.id}`}
        />
      ) : null}

      {insights ? (
        <AutomationRunStats state={insights.state} copy={insights.spec.stats} />
      ) : null}
    </div>
  );
}
