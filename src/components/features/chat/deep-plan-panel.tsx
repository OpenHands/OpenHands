import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, Lock } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useConversationStore } from "#/stores/conversation-store";
import {
  DEEP_PLAN_PHASES,
  getDeepPlanPhase,
  type DeepPlanPhaseId,
} from "#/utils/deep-plan";
import { validateDocumentChain } from "#/utils/deep-plan-reference";
import type { DeepPlanDocuments } from "#/utils/deep-plan-reference";
import {
  DEEP_PLAN_PHASE_LABEL_KEY,
  makeRefIssueMessage,
} from "#/utils/deep-plan-messages";
import {
  canEnterPhase,
  deepPlanUnavailableDocuments,
  isPhaseConfirmed,
  type ConfirmFailure,
} from "#/utils/deep-plan-machine";
import { cn } from "#/utils/utils";

/**
 * The inputs a checkpoint validation depends on: the active phase and the
 * documents as they were when Confirm ran. A failure is only meaningful for the
 * revision it was produced from.
 */
interface DeepPlanRevision {
  activePhase: DeepPlanPhaseId | null;
  documents: DeepPlanDocuments;
}

const sameDeepPlanRevision = (
  a: DeepPlanRevision,
  b: DeepPlanRevision,
): boolean => a.activePhase === b.activePhase && a.documents === b.documents;

/**
 * The phase rail plus the checkpoint for the active phase. The rail is the
 * only place the user learns why a later phase is unreachable, so locked
 * phases render as disabled rather than hidden.
 */
export function DeepPlanPanel() {
  const { t } = useTranslation("openhands");
  const {
    deepPlan,
    setDeepPlanPhase,
    confirmDeepPlanPhase,
    startDeepPlan,
    retryDeepPlanDocumentRestore,
  } = useConversationStore();
  // Tie the checkpoint failure to the exact revision it described: the phase
  // and the documents snapshot at confirm time. Moving to another phase or
  // editing the documents changes the revision, so the stale error simply stops
  // rendering — no effect needed to clear it, and nothing to miss when the
  // revision changes between a render and its effect.
  const [failure, setFailure] = useState<{
    revision: DeepPlanRevision;
    failure: ConfirmFailure;
  } | null>(null);

  const activePhase = deepPlan.activePhase;

  // Validate only up to the active phase, matching the checkpoint. A citation
  // in a document the user has not reached yet is not actionable from here, and
  // showing it would contradict the checkpoint that lets them continue.
  const report = useMemo(
    () => validateDocumentChain(deepPlan.documents, activePhase ?? undefined),
    [deepPlan.documents, activePhase],
  );

  const revision: DeepPlanRevision = {
    activePhase,
    documents: deepPlan.documents,
  };

  const handleConfirm = () => {
    if (!activePhase) return;
    const result = confirmDeepPlanPhase(activePhase);
    setFailure(
      result.ok || !result.failure
        ? null
        : { revision, failure: result.failure },
    );
  };

  // Only show the failure while the revision still matches what was confirmed.
  const visibleFailure =
    failure && sameDeepPlanRevision(failure.revision, revision)
      ? failure.failure
      : null;

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

  // Documents the persisted chain vouches for but that are not in memory yet. A
  // reload or an in-app conversation switch hydrates the slim state (bodies
  // empty), so until the disk re-read lands every upstream citation would look
  // like a missing document. Surface it and hold the checkpoint rather than
  // validating a half-restored chain.
  const unavailable = deepPlanUnavailableDocuments(deepPlan, activePhase);
  const restoreBlocked = unavailable.pending.length > 0;
  // A phase whose disk re-read was rejected leaves the checkpoint unable to
  // validate it. Offer a Retry rather than an automatic re-read, which would
  // loop forever when the file is genuinely gone.
  const showRestoreRetry =
    unavailable.failed.length > 0 || visibleFailure?.kind === "restore-failed";
  const formatPhases = (phases: DeepPlanPhaseId[]) =>
    phases
      .map(
        (candidate) =>
          getDeepPlanPhase(candidate).outputFile ??
          t(DEEP_PLAN_PHASE_LABEL_KEY[candidate]),
      )
      .join(", ");

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

        {restoreBlocked && (
          <Typography.Text className="text-xs" testId="deep-plan-restoring">
            {t(I18nKey.DEEP_PLAN$CONFIRM_RESTORING, {
              document: formatPhases(unavailable.pending),
            })}
          </Typography.Text>
        )}

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

        {visibleFailure && (
          <Typography.Text className="text-xs" testId="deep-plan-error">
            {visibleFailure.kind === "blocked"
              ? t(I18nKey.DEEP_PLAN$CONFIRM_BLOCKED, {
                  phase: t(DEEP_PLAN_PHASE_LABEL_KEY[visibleFailure.phase]),
                })
              : visibleFailure.kind === "missing-output"
                ? t(I18nKey.DEEP_PLAN$CONFIRM_MISSING_OUTPUT, {
                    document:
                      getDeepPlanPhase(visibleFailure.phase).outputFile ??
                      t(DEEP_PLAN_PHASE_LABEL_KEY[visibleFailure.phase]),
                  })
                : visibleFailure.kind === "restoring"
                  ? t(I18nKey.DEEP_PLAN$CONFIRM_RESTORING, {
                      document: formatPhases(visibleFailure.phases),
                    })
                  : visibleFailure.kind === "restore-failed"
                    ? t(I18nKey.DEEP_PLAN$CONFIRM_RESTORE_FAILED, {
                        document: formatPhases(visibleFailure.phases),
                      })
                    : `${makeRefIssueMessage(t, visibleFailure.issue)}${
                        visibleFailure.extraCount > 0
                          ? t(I18nKey.DEEP_PLAN$CONFIRM_MORE_ISSUES, {
                              count: visibleFailure.extraCount,
                            })
                          : ""
                      }`}
          </Typography.Text>
        )}

        {showRestoreRetry && (
          <BrandButton
            type="button"
            variant="secondary"
            onClick={() => {
              // Retrying re-reads the document, so the recorded failure stops
              // being true. The revision only tracks phase + documents, not
              // restore status, so the stale error would otherwise keep
              // rendering throughout the new read.
              setFailure(null);
              retryDeepPlanDocumentRestore();
            }}
            testId="deep-plan-restore-retry"
            className="min-w-40 justify-center px-6"
          >
            {t(I18nKey.DEEP_PLAN$RESTORE_RETRY)}
          </BrandButton>
        )}

        <BrandButton
          type="button"
          variant="secondary"
          onClick={handleConfirm}
          testId="deep-plan-confirm"
          isDisabled={restoreBlocked}
          className="min-w-40 justify-center px-6"
        >
          {t(I18nKey.COMMON$DEEP_PLAN_CONFIRM)}
        </BrandButton>
      </div>
    </div>
  );
}
