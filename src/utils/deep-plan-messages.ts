import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import { getDeepPlanPhase, type DeepPlanPhaseId } from "#/utils/deep-plan";
import type { RefIssue } from "#/utils/deep-plan-reference";

/** Locale key for a phase's display name. */
export const DEEP_PLAN_PHASE_LABEL_KEY: Record<DeepPlanPhaseId, I18nKey> = {
  analysis: I18nKey.DEEP_PLAN$PHASE_ANALYSIS,
  requirements: I18nKey.DEEP_PLAN$PHASE_REQUIREMENTS,
  database: I18nKey.DEEP_PLAN$PHASE_DATABASE,
  backend: I18nKey.DEEP_PLAN$PHASE_BACKEND,
  frontend: I18nKey.DEEP_PLAN$PHASE_FRONTEND,
  tasks: I18nKey.DEEP_PLAN$PHASE_TASKS,
  implementation: I18nKey.DEEP_PLAN$PHASE_IMPLEMENTATION,
};

/**
 * English phase guidance for the planner's system prompt. Resolved at
 * `lng: "en"` on purpose: the planner's directive must be stable regardless of
 * the UI locale (an agent prompt is not user copy), while the panel keeps
 * showing `t(instructionKey)` in the active locale. `t` is passed in so this
 * module does not import a particular i18n instance.
 */
export function deepPlanGuidance(
  t: TFunction<"openhands">,
  phase: DeepPlanPhaseId,
): string {
  return t(getDeepPlanPhase(phase).instructionKey, { lng: "en" });
}

/**
 * The document an issue is reported against: its output filename when the
 * phase produces one, otherwise the phase's display name. Localized so the
 * checkpoint error follows the active locale.
 */
function documentName(
  t: TFunction<"openhands">,
  from: DeepPlanPhaseId,
): string {
  const phase = getDeepPlanPhase(from);
  return phase.outputFile ?? t(DEEP_PLAN_PHASE_LABEL_KEY[from]);
}

/**
 * Localized reason a citation was rejected. `t` is passed in rather than
 * imported so the message follows the caller's i18n instance.
 */
export function makeRefIssueMessage(
  t: TFunction<"openhands">,
  issue: RefIssue,
): string {
  const document = documentName(t, issue.from);
  switch (issue.reason) {
    case "dangling":
      return t(I18nKey.DEEP_PLAN$ISSUE_DANGLING, {
        document,
        ref: issue.ref,
      });
    case "not-upstream":
      return t(I18nKey.DEEP_PLAN$ISSUE_NOT_UPSTREAM, {
        document,
        ref: issue.ref,
      });
    case "missing-document":
      return t(I18nKey.DEEP_PLAN$ISSUE_MISSING_DOCUMENT, {
        document,
        ref: issue.ref,
      });
  }
}
