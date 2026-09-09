import { useTranslation } from "react-i18next";
import { LOOPS_PATH, loopRunPath } from "#/api/loop-service/loop-constants";
import type { LoopDefinition } from "#/api/loop-service/loop-types";
import { automationIconActionButtonClassName } from "#/components/features/automations/automation-action-button-classes";
import { CostText } from "#/components/shared/cost-text";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useNavigation } from "#/context/navigation-context";
import { useFireLoopTrigger, useLoopRuns } from "#/hooks/query/use-loops";
import PlayIcon from "#/icons/play.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardPillClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

function runStatusKey(status: string): I18nKey {
  if (status === "passed") return I18nKey.FEATURE_DEV$STATUS_PASSED;
  if (status === "failed") return I18nKey.FEATURE_DEV$STATUS_FAILED;
  if (status === "aborted") return I18nKey.FEATURE_DEV$STATUS_ABORTED;
  if (status === "awaiting_input") return I18nKey.FEATURE_DEV$STATUS_PAUSED;
  if (status === "pending") return I18nKey.FEATURE_DEV$STATUS_PENDING;
  return I18nKey.FEATURE_DEV$STATUS_RUNNING;
}

interface LoopDefinitionCardProps {
  definition: LoopDefinition;
  runTriggerId: string | null;
}

export function LoopDefinitionCard({
  definition,
  runTriggerId,
}: LoopDefinitionCardProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const runsQuery = useLoopRuns(definition.id);
  const fireTrigger = useFireLoopTrigger();
  const last = runsQuery.data?.[runsQuery.data.length - 1];

  const openManagement = () => navigate(LOOPS_PATH);

  return (
    <div
      role="link"
      tabIndex={0}
      data-testid={`loop-definition-${definition.id}`}
      onClick={openManagement}
      onKeyDown={(event) => {
        if (event.key === "Enter") openManagement();
      }}
      className={cn(
        "group relative flex min-w-0 flex-col overflow-hidden p-4 text-left",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
      )}
    >
      <header className="flex h-8 items-center justify-between gap-3">
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold leading-none text-[var(--oh-foreground)]">
          {definition.name}
        </h3>
        <StyledTooltip content={t(I18nKey.LOOPS$RUN_NOW)} placement="top">
          <button
            type="button"
            data-testid={`loop-definition-run-now-${definition.id}`}
            aria-label={t(I18nKey.LOOPS$RUN_NOW)}
            disabled={!runTriggerId || fireTrigger.isPending}
            onClick={(event) => {
              event.stopPropagation();
              if (!runTriggerId) return;
              fireTrigger.mutate(runTriggerId, {
                onError: () => displayErrorToast(t(I18nKey.ERROR$GENERIC)),
              });
            }}
            className={automationIconActionButtonClassName}
          >
            <PlayIcon className="size-4 shrink-0" aria-hidden />
          </button>
        </StyledTooltip>
      </header>

      {definition.stages.length > 0 ? (
        <div className="mt-3 flex min-w-0 flex-wrap gap-1">
          {definition.stages.map((stage) => (
            <span
              key={stage.name}
              className={cn(extensionModuleCardPillClassName, "text-white")}
            >
              {stage.name}
            </span>
          ))}
        </div>
      ) : null}

      <div className="mt-3 flex items-center justify-between gap-2">
        {last ? (
          <button
            type="button"
            data-testid={`loop-definition-last-run-${definition.id}`}
            className={cn(extensionModuleCardPillClassName, "text-white")}
            onClick={(event) => {
              event.stopPropagation();
              navigate(loopRunPath(last.id));
            }}
          >
            {t(runStatusKey(last.status))}
          </button>
        ) : (
          <span className="text-xs text-tertiary-light">
            {t(I18nKey.LOOPS$LAST_RUN)}
          </span>
        )}
        {last ? (
          <span className="text-xs text-tertiary-light">
            {t(I18nKey.LOOPS$COST)}
            <span className="ml-1 tabular-nums text-white">
              <CostText
                amount={last.total_cost_usd}
                at={last.updated_at ?? last.created_at}
              />
            </span>
          </span>
        ) : null}
      </div>
    </div>
  );
}
