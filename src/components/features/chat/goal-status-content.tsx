import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import CheckCircle from "#/icons/check-circle-solid.svg?react";
import XCircle from "#/icons/x-circle-solid.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { GoalStatus } from "#/types/agent-server/core/events/conversation-state-event";
import { useGoalStore } from "#/stores/goal-store";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  pauseConversation,
  resumeGoal,
  stopGoal,
} from "#/hooks/mutation/conversation-mutation-utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { cn } from "#/utils/utils";
import {
  chatInputPillButtonClassName,
  formControlDisabledClassName,
} from "#/utils/form-control-classes";
import { GenericEventMessage } from "./generic-event-message";

const STATUS_LABEL_KEY: Record<GoalStatus["status"], I18nKey> = {
  running: I18nKey.GOAL$STATUS_RUNNING,
  complete: I18nKey.GOAL$STATUS_COMPLETE,
  capped: I18nKey.GOAL$STATUS_CAPPED,
  interrupted: I18nKey.GOAL$STATUS_INTERRUPTED,
};

/**
 * Goal-status row: objective, round count, status word, the judge's score, the
 * judge's "missing" note (expandable), and an indicator — spinner while running,
 * green check when complete, muted cross when it ends without completing
 * (capped/interrupted).
 *
 * Also exposes the loop controls at the end of the row: a Stop button while the
 * loop is active and a Resume button once it is interrupted. Stop both cancels
 * the loop (`stopGoal`) and interrupts the conversation, because the backend's
 * stop deliberately leaves the in-flight agent turn running.
 *
 * Used in two places: the live bottom banner (GoalStatusBanner) while a loop is
 * active, and inline in the message timeline for the terminal status, so a
 * finished `/goal` settles into the conversation. Because the inline copy mounts
 * fresh once terminal, `initiallyExpanded={!active}` expands the note there
 * without any re-mount trickery.
 */
export function GoalStatusContent({ status }: { status: GoalStatus }) {
  const { t } = useTranslation("openhands");
  const { conversationId } = useOptionalConversationId();
  // Whether *some* goal loop is currently live for this conversation. Used to
  // hide a stale inline Resume button once the user has resumed (or started a
  // new goal): resuming again would just 409 on the backend.
  const goalActive = useGoalStore((s) =>
    conversationId
      ? Boolean(s.statusByConversation[conversationId]?.active)
      : false,
  );
  const [pending, setPending] = useState(false);
  const {
    active,
    objective,
    iteration,
    max_iterations: maxIterations,
    verdict,
  } = status;
  const scorePct = verdict ? Math.round(verdict.score * 100) : null;
  const details = verdict?.missing
    ? t(I18nKey.GOAL$MISSING, { missing: verdict.missing })
    : "";

  const runAction = async (
    action: (id: string) => Promise<void>,
    failKey: I18nKey,
  ) => {
    if (!conversationId || pending) return;
    setPending(true);
    try {
      await action(conversationId);
    } catch (err) {
      displayErrorToast(
        err instanceof Error && err.message ? err.message : t(failKey),
      );
    } finally {
      setPending(false);
    }
  };

  // stopGoal only cancels the loop; the backend leaves the current agent turn
  // running, so interrupt the conversation too to actually halt it.
  const handleStop = () =>
    runAction(async (id) => {
      await stopGoal(id);
      await pauseConversation(id);
    }, I18nKey.GOAL$STOP_FAILED);

  const handleResume = () => runAction(resumeGoal, I18nKey.GOAL$RESUME_FAILED);

  let actionButton: ReactNode = null;
  if (conversationId && active) {
    actionButton = (
      <button
        type="button"
        data-testid="goal-stop"
        disabled={pending}
        onClick={handleStop}
        className={cn(
          chatInputPillButtonClassName,
          formControlDisabledClassName,
        )}
      >
        {t(I18nKey.GOAL$STOP)}
      </button>
    );
  } else if (conversationId && status.status === "interrupted" && !goalActive) {
    actionButton = (
      <button
        type="button"
        data-testid="goal-resume"
        disabled={pending}
        onClick={handleResume}
        className={cn(
          chatInputPillButtonClassName,
          formControlDisabledClassName,
        )}
      >
        {t(I18nKey.GOAL$RESUME)}
      </button>
    );
  }

  return (
    <div data-testid="goal-status" className="flex flex-col w-full">
      <GenericEventMessage
        title={
          <span className="flex items-center gap-2 flex-wrap">
            <span className="opacity-60">{t(I18nKey.GOAL$PREFIX)}</span>
            <span>{objective}</span>
            <span className="opacity-60">
              {t(I18nKey.GOAL$ROUND, { iteration, max: maxIterations })}
            </span>
            <span>{t(STATUS_LABEL_KEY[status.status])}</span>
            {scorePct !== null && (
              <span className="opacity-60">
                {t(I18nKey.GOAL$SCORE, { score: scorePct })}
              </span>
            )}
            {active ? (
              <span
                data-testid="goal-spinner"
                className="inline-block w-3.5 h-3.5 ml-1 rounded-full border-2 border-transparent border-t-[var(--oh-border-input)] animate-spin"
              />
            ) : status.status === "complete" ? (
              <span data-testid="goal-done" className="inline-flex ml-1">
                <CheckCircle className="w-3.5 h-3.5 fill-success" />
              </span>
            ) : (
              <span data-testid="goal-ended" className="inline-flex ml-1">
                <XCircle className="w-3.5 h-3.5 fill-[var(--oh-muted)]" />
              </span>
            )}
          </span>
        }
        titleTrailing={actionButton}
        details={details}
        initiallyExpanded={!active}
        chevronPosition="before"
      />
    </div>
  );
}
