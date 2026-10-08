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
  getDeepPlanPhase,
  nextDeepPlanPhase,
} from "#/utils/deep-plan";
import {
  validateDocumentChain,
  type DeepPlanDocuments,
  type RefIssue,
} from "#/utils/deep-plan-reference";

export interface DeepPlanState {
  /** Phase the user is working in; `null` before the mode is started. */
  activePhase: DeepPlanPhaseId | null;
  /** Phases whose checkpoint the user has passed, in order. */
  confirmed: DeepPlanPhaseId[];
  /** Document contents, keyed by the phase that produced them. */
  documents: DeepPlanDocuments;
  /**
   * Fingerprint of each document's bytes, keyed by phase. Persisted (unlike the
   * bodies) so that after a reload `setDeepPlanDocument` can tell a history
   * replay of the same bytes from a real edit and keep the confirmations the
   * persisted chain vouches for. Optional because a state built before the
   * hashes were recorded still has to typecheck.
   */
  documentHashes?: DeepPlanDocuments;
  /**
   * Phases whose document could not be re-read from disk during a restore
   * (planner gone, file deleted under it). Tracked explicitly rather than
   * silently left empty so the checkpoint can name the missing document instead
   * of reporting it as an ordinary dangling/missing citation.
   */
  restoreFailed?: DeepPlanPhaseId[];
}

/**
 * The phase machine as persisted to localStorage. Document *bodies* are
 * deliberately omitted: they are re-read from disk on reload via history
 * replay, and keeping full documents in the consolidated blob risks hitting the
 * localStorage quota. The per-document hashes are kept so rehydrate detection
 * survives the reload.
 */
export interface PersistedDeepPlanState {
  activePhase: DeepPlanPhaseId | null;
  confirmed: DeepPlanPhaseId[];
  documentHashes?: DeepPlanDocuments;
}

/**
 * Stable non-cryptographic fingerprint (djb2, 32-bit) of a document's bytes.
 * Used only to compare two revisions for equality, never as a security
 * primitive, so a cheap hash is appropriate.
 */
export function hashDeepPlanDocument(content: string): string {
  let hash = 5381;
  for (let i = 0; i < content.length; i += 1) {
    hash = (hash * 33) ^ content.charCodeAt(i);
  }
  return (hash >>> 0).toString(36);
}

/** Restore the in-memory machine from its persisted form, bodies left empty. */
export function hydrateDeepPlanState(
  persisted: PersistedDeepPlanState | undefined,
): DeepPlanState {
  if (!persisted) return EMPTY_DEEP_PLAN_STATE;
  return {
    activePhase: persisted.activePhase,
    confirmed: persisted.confirmed,
    documents: {},
    documentHashes: persisted.documentHashes ?? {},
    restoreFailed: [],
  };
}

/**
 * Documents the persisted chain vouches for (a recorded hash) but whose bytes
 * are not in memory. After a reload every document is here until the disk
 * re-read lands; a checkpoint run in that window would otherwise see each
 * upstream citation as a missing document. Derived rather than stored so it
 * cannot be left stale by a lifecycle race.
 *
 * `pending` are still being re-read (or have not been attempted yet); `failed`
 * had their re-read rejected (planner/file gone), so the chain genuinely cannot
 * be validated for them. Scoped to `throughPhase` so an unavailable document
 * beyond the checkpoint does not gate it.
 */
export interface UnavailableDeepPlanDocuments {
  pending: DeepPlanPhaseId[];
  failed: DeepPlanPhaseId[];
}

export function deepPlanUnavailableDocuments(
  state: DeepPlanState,
  throughPhase?: DeepPlanPhaseId,
): UnavailableDeepPlanDocuments {
  const hashes = state.documentHashes ?? {};
  const failedSet = new Set(state.restoreFailed ?? []);
  const lastIndex = throughPhase
    ? DEEP_PLAN_PHASE_IDS.indexOf(throughPhase)
    : DEEP_PLAN_PHASE_IDS.length - 1;
  const pending: DeepPlanPhaseId[] = [];
  const failed: DeepPlanPhaseId[] = [];
  for (const phase of DEEP_PLAN_PHASE_IDS) {
    if (DEEP_PLAN_PHASE_IDS.indexOf(phase) > lastIndex) continue;
    if (hashes[phase] === undefined || state.documents[phase] !== undefined) {
      continue;
    }
    if (failedSet.has(phase)) failed.push(phase);
    else pending.push(phase);
  }
  return { pending, failed };
}

/** Drop the document bodies for storage; keep phase, confirmations and hashes. */
export function toPersistedDeepPlan(
  state: DeepPlanState,
): PersistedDeepPlanState {
  return {
    activePhase: state.activePhase,
    confirmed: state.confirmed,
    documentHashes: state.documentHashes ?? {},
  };
}

export const EMPTY_DEEP_PLAN_STATE: DeepPlanState = {
  activePhase: null,
  confirmed: [],
  documents: {},
  documentHashes: {},
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

/**
 * Why a checkpoint was refused. Returned as structured data rather than a
 * pre-rendered string so the caller localizes it; this module stays free of
 * `t()` and of any particular locale.
 */
export type ConfirmFailure =
  /** An earlier phase is still unconfirmed and must be confirmed first. */
  | { kind: "blocked"; phase: DeepPlanPhaseId }
  /**
   * The phase must produce a document but `documents[phase]` is still absent.
   * Confirming without it would advance the chain on no evidence — the very
   * document the checkpoint is meant to vouch for. Pure-conversation phases
   * (`outputFile === null`) are unaffected.
   */
  | { kind: "missing-output"; phase: DeepPlanPhaseId }
  /**
   * A document this checkpoint needs is vouched for by the persisted chain but
   * its bytes are not in memory yet — the reload re-read is still in flight, or
   * it failed (see `restoreFailed`). Validating now would treat the recorded
   * hash as if it were a body and report every upstream citation as a missing
   * document; name the unavailable phases instead.
   */
  | { kind: "restoring"; phases: DeepPlanPhaseId[] }
  /**
   * A document this checkpoint needs had its reload re-read rejected — the
   * planner or file is gone. Distinct from `restoring` so the panel can tell
   * the user the chain cannot be validated rather than that it is arriving.
   */
  | { kind: "restore-failed"; phases: DeepPlanPhaseId[] }
  /** The reference chain is invalid; `issue` names the offending citation. */
  | { kind: "invalid-chain"; issue: RefIssue; extraCount: number };

export type ConfirmResult =
  | { ok: true; state: DeepPlanState }
  | { ok: false; failure: ConfirmFailure };

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
    return { ok: false, failure: { kind: "blocked", phase: blocking! } };
  }

  // A reload re-reads the persisted documents from disk. Until it lands, the
  // bodies are absent but the chain still vouches for them, so validating now
  // would report every upstream citation as a missing document. Refuse and name
  // the phases rather than treating a recorded hash as a body.
  const pendingRestore = deepPlanUnavailableDocuments(state, phase);
  if (pendingRestore.pending.length > 0) {
    return {
      ok: false,
      failure: { kind: "restoring", phases: pendingRestore.pending },
    };
  }
  if (pendingRestore.failed.length > 0) {
    return {
      ok: false,
      failure: { kind: "restore-failed", phases: pendingRestore.failed },
    };
  }

  // A phase that must produce a document cannot be confirmed before that
  // document exists: there would be nothing to validate, so the checkpoint
  // would advance the chain on no evidence. Pure-conversation phases
  // (`outputFile === null`) legitimately have no document to require.
  if (getDeepPlanPhase(phase).outputFile && !state.documents[phase]) {
    return { ok: false, failure: { kind: "missing-output", phase } };
  }

  // Validate the chain only through the phase being confirmed. Later documents
  // are still unreviewed at this checkpoint, so their citations must not gate
  // it — they get their own checkpoint when the user reaches them.
  const report = validateDocumentChain(state.documents, phase);
  if (!report.ok) {
    const [first] = report.issues;
    return {
      ok: false,
      failure: {
        kind: "invalid-chain",
        issue: first,
        extraCount: report.issues.length - 1,
      },
    };
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
