/**
 * Phase state machine for Deep Planning (#17819).
 *
 * Pure transitions so the gating rules ("a phase cannot start until the
 * previous one is confirmed", "confirm runs the reference validator first")
 * are unit-testable without React or a store. The conversation store holds the
 * one authoritative copy of this state and is the only writer.
 */

import {
  DEEP_PLAN_PHASE_IDS,
  type DeepPlanPhaseId,
  nextDeepPlanPhase,
} from "#/utils/deep-plan";
import {
  describeRefIssue,
  validateDocumentChain,
  type DeepPlanDocuments,
} from "#/utils/deep-plan-reference";

export interface DeepPlanState {
  /** Phase the user is working in; `null` before the mode is started. */
  activePhase: DeepPlanPhaseId | null;
  /** Phases whose checkpoint the user has passed, in order. */
  confirmed: DeepPlanPhaseId[];
  /** Document contents, keyed by the phase that produced them. */
  documents: DeepPlanDocuments;
}

export const EMPTY_DEEP_PLAN_STATE: DeepPlanState = {
  activePhase: null,
  confirmed: [],
  documents: {},
};

export const startDeepPlan = (): DeepPlanState => ({
  ...EMPTY_DEEP_PLAN_STATE,
  activePhase: DEEP_PLAN_PHASE_IDS[0],
});

export const isPhaseConfirmed = (
  state: DeepPlanState,
  phase: DeepPlanPhaseId,
): boolean => state.confirmed.includes(phase);

/**
 * Every phase before `phase` is confirmed. The first phase is always
 * enterable; a phase already confirmed stays enterable so the user can go back
 * and revise an earlier document without losing the chain.
 */
export function canEnterPhase(
  state: DeepPlanState,
  phase: DeepPlanPhaseId,
): boolean {
  const index = DEEP_PLAN_PHASE_IDS.indexOf(phase);
  if (index <= 0) return true;
  return DEEP_PLAN_PHASE_IDS.slice(0, index).every((earlier) =>
    isPhaseConfirmed(state, earlier),
  );
}

/**
 * Drop `phase`'s confirmation and every later one. A checkpoint only vouches
 * for the documents it saw; once an upstream document is rewritten the
 * confirmations built on it are no longer evidence that the chain is valid, so
 * they must be re-earned rather than silently kept.
 */
export function invalidateFrom(
  state: DeepPlanState,
  phase: DeepPlanPhaseId,
): DeepPlanState {
  const index = DEEP_PLAN_PHASE_IDS.indexOf(phase);
  const confirmed = state.confirmed.filter(
    (confirmedPhase) => DEEP_PLAN_PHASE_IDS.indexOf(confirmedPhase) < index,
  );
  if (confirmed.length === state.confirmed.length) return state;
  // The active phase may have been one of the invalidated ones; send the user
  // back to the edited phase so they re-walk the chain from where it broke.
  const activeIndex = state.activePhase
    ? DEEP_PLAN_PHASE_IDS.indexOf(state.activePhase)
    : -1;
  const activePhase = activeIndex >= index ? phase : state.activePhase;
  return { ...state, confirmed, activePhase };
}

export type ConfirmResult =
  | { ok: true; state: DeepPlanState }
  | { ok: false; error: string };

/**
 * Pass the checkpoint for `phase`: validate the reference chain, then mark the
 * phase confirmed and move the active phase forward. A validation failure
 * blocks the transition and names the offending citation.
 */
export function confirmPhase(
  state: DeepPlanState,
  phase: DeepPlanPhaseId,
): ConfirmResult {
  if (!canEnterPhase(state, phase)) {
    const blocking = DEEP_PLAN_PHASE_IDS.slice(
      0,
      DEEP_PLAN_PHASE_IDS.indexOf(phase),
    ).find((earlier) => !isPhaseConfirmed(state, earlier));
    return {
      ok: false,
      error: `Confirm ${blocking} before continuing.`,
    };
  }

  // Validate the chain only through the phase being confirmed. Later documents
  // are still unreviewed at this checkpoint, so their citations must not gate
  // it — they get their own checkpoint when the user reaches them.
  const report = validateDocumentChain(state.documents, phase);
  if (!report.ok) {
    const [first] = report.issues;
    const extra =
      report.issues.length > 1 ? ` (+${report.issues.length - 1} more)` : "";
    return { ok: false, error: `${describeRefIssue(first)}${extra}` };
  }

  const confirmed = isPhaseConfirmed(state, phase)
    ? state.confirmed
    : [...state.confirmed, phase];

  // The final phase has no successor: keep it active rather than dropping the
  // user back to the "not started" empty state.
  const nextActive = isPhaseConfirmed(state, phase)
    ? state.activePhase
    : (nextDeepPlanPhase(phase) ?? phase);

  return {
    ok: true,
    state: { ...state, confirmed, activePhase: nextActive },
  };
}
