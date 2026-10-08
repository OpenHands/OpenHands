import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, Lock } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useConversationStore } from "#/stores/conversation-store";
import { DEEP_PLAN_PHASES, getDeepPlanPhase } from "#/utils/deep-plan";
import { validateDocumentChain } from "#/utils/deep-plan-reference";
import {
  DEEP_PLAN_PHASE_LABEL_KEY,
  makeRefIssueMessage,
} from "#/utils/deep-plan-messages";
import {
  canEnterPhase,
  isPhaseConfirmed,
  type ConfirmFailure,
} from "#/utils/deep-plan-machine";
import { cn } from "#/utils/utils";

/**
 * The phase rail plus the checkpoint for the active phase. The rail is the
 * only place the user learns why a later phase is unreachable, so locked
 * phases render as disabled rather than hidden.
 */
export function DeepPlanPanel() {
  const { t } = useTranslation("openhands");
  const { deepPlan, setDeepPlanPhase, confirmDeepPlanPhase, startDeepPlan } =
    useConversationStore();
  const [failure, setFailure] = useState<ConfirmFailure | null>(null);

  const activePhase = deepPlan.activePhase;

  // A checkpoint error describes the document as it was when Confirm was
  // pressed. Moving to another phase or editing the documents makes it stale,
  // so clear it rather than leaving the old error on screen.
  useEffect(() => {
    setFailure(null);
  }, [activePhase, deepPlan.documents]);

  // Validate only up to the active phase, matching the checkpoint. A citation
  // in a document the user has not reached yet is not actionable from here, and
  // showing it would contradict the checkpoint that lets them continue.
  const report = useMemo(
    () => validateDocumentChain(deepPlan.documents, activePhase ?? undefined),
    [deepPlan.documents, activePhase],
  );

  const handleConfirm = () => {
    if (!activePhase) return;
    const result = confirmDeepPlanPhase(activePhase);
    setFailure(result.ok ? null : (result.failure ?? null));
  };

  if (!activePhase) {
    return (
      <div className="flex flex-col gap-3 p-4">
        <Typography.Text className="text-sm">
          {t(I18nKey.DEEP_PLAN$EMPTY_MESSAGE)}
        </Typography.Text>
        <BrandButton
          type="button"
          variant="secondary"
          onClick={() => startDeepPlan()}
          className="min-w-40 justify-center px-6"
        >
          {t(I18nKey.DEEP_PLAN$START)}
        </BrandButton>
      </div>
    );
  }

  const phase = getDeepPlanPhase(activePhase);

  return (
    <div className="flex flex-col gap-4 p-4">
      <Typography.Text className="text-sm font-medium">
        {t(I18nKey.DEEP_PLAN$TITLE)}
      </Typography.Text>

      <ol className="flex flex-col gap-1" data-testid="deep-plan-phase-rail">
        {DEEP_PLAN_PHASES.map((candidate) => {
          const confirmed = isPhaseConfirmed(deepPlan, candidate.id);
          const enterable = canEnterPhase(deepPlan, candidate.id);
          const isActive = candidate.id === activePhase;
          return (
            <li key={candidate.id}>
              <button
                type="button"
                disabled={!enterable}
                data-testid={`deep-plan-phase-${candidate.id}`}
                data-state={
                  confirmed ? "confirmed" : isActive ? "active" : "locked"
                }
                onClick={() => setDeepPlanPhase(candidate.id)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-sm",
                  isActive && "bg-interactive-hover",
                  !enterable && "cursor-not-allowed opacity-50",
                )}
              >
                {confirmed ? (
                  <Check aria-hidden width={14} height={14} />
                ) : enterable ? null : (
                  <Lock aria-hidden width={14} height={14} />
                )}
                {t(DEEP_PLAN_PHASE_LABEL_KEY[candidate.id])}
              </button>
            </li>
          );
        })}
      </ol>

      <div className="flex flex-col gap-2 border-t border-border pt-3">
        <Typography.Text className="text-xs text-muted">
          {phase.outputFile ?? t(I18nKey.DEEP_PLAN$PHASE_ANALYSIS)}
        </Typography.Text>
        <Typography.Text className="text-xs whitespace-pre-line">
          {t(phase.instructionKey)}
        </Typography.Text>

        {report.issues.length > 0 && (
          <ul className="flex flex-col gap-1" data-testid="deep-plan-issues">
            {report.issues.map((issue) => (
              <li key={`${issue.from}-${issue.ref}`} className="text-xs">
                {makeRefIssueMessage(t, issue)}
              </li>
            ))}
          </ul>
        )}

        {report.uncovered.length > 0 && (
          <Typography.Text className="text-xs" testId="deep-plan-uncovered">
            {t(I18nKey.COMMON$DEEP_PLAN_UNCOVERED, {
              sections: report.uncovered.join(", "),
            })}
          </Typography.Text>
        )}

        {failure && (
          <Typography.Text className="text-xs" testId="deep-plan-error">
            {failure.kind === "blocked"
              ? t(I18nKey.DEEP_PLAN$CONFIRM_BLOCKED, {
                  phase: t(DEEP_PLAN_PHASE_LABEL_KEY[failure.phase]),
                })
              : `${makeRefIssueMessage(t, failure.issue)}${
                  failure.extraCount > 0
                    ? t(I18nKey.DEEP_PLAN$CONFIRM_MORE_ISSUES, {
                        count: failure.extraCount,
                      })
                    : ""
                }`}
          </Typography.Text>
        )}

        <BrandButton
          type="button"
          variant="secondary"
          onClick={handleConfirm}
          testId="deep-plan-confirm"
          className="min-w-40 justify-center px-6"
        >
          {t(I18nKey.COMMON$DEEP_PLAN_CONFIRM)}
        </BrandButton>
      </div>
    </div>
  );
}
