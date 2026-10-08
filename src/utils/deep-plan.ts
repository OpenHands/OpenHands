/**
 * Deep Planning mode (#17819): a gated chain of documents where every section
 * cites the upstream sections it derives from, so the final implementation is
 * traceable back to a requirement.
 *
 * This module is the single source of truth for the phase order, the reference
 * vocabulary and the per-phase prompt. The reference *validation* lives in
 * `deep-plan-reference.ts` so it stays pure and unit-testable.
 */

import { I18nKey } from "#/i18n/declaration";

export type DeepPlanPhaseId =
  | "analysis"
  | "requirements"
  | "database"
  | "backend"
  | "frontend"
  | "tasks"
  | "implementation";

/** The label a document's sections carry for downstream citation. */
export type DeepPlanLabel = "Req" | "DB" | "BE" | "FE";

export interface DeepPlanPhase {
  id: DeepPlanPhaseId;
  /** File the planner writes for this phase. `null` for pure-conversation phases. */
  outputFile: string | null;
  /** Labels this phase's sections must cite; must exist upstream. */
  cites: readonly DeepPlanLabel[];
  /** Label this phase's sections carry, so downstream phases can cite them. */
  defines: DeepPlanLabel | null;
  /**
   * I18n key for the phase-specific workflow guidance. Two consumers:
   *
   * - The Planner panel renders it through `t()` so it follows the active
   *   locale — that is the copy the user reads.
   * - `deepPlanGuidance` (deep-plan-messages.ts) resolves it at `lng: "en"` and
   *   the per-phase planner wiring (#18104, Route A) appends it to that phase's
   *   planner system prompt, so the planner is actually driven to produce the
   *   phase's document. The prompt stays locale-independent on purpose.
   */
  instructionKey: I18nKey;
}

/**
 * Phase order is the contract: a phase cannot start until the previous one is
 * confirmed, and a reference is only valid if it points at an *earlier* phase.
 */
export const DEEP_PLAN_PHASES: readonly DeepPlanPhase[] = [
  {
    id: "analysis",
    outputFile: null,
    cites: [],
    defines: null,
    instructionKey: I18nKey.DEEP_PLAN$INSTRUCTION_ANALYSIS,
  },
  {
    id: "requirements",
    outputFile: "requirements.md",
    cites: [],
    defines: "Req",
    instructionKey: I18nKey.DEEP_PLAN$INSTRUCTION_REQUIREMENTS,
  },
  {
    id: "database",
    outputFile: "database-design.md",
    cites: ["Req"],
    defines: "DB",
    instructionKey: I18nKey.DEEP_PLAN$INSTRUCTION_DATABASE,
  },
  {
    id: "backend",
    outputFile: "backend-design.md",
    cites: ["Req", "DB"],
    defines: "BE",
    instructionKey: I18nKey.DEEP_PLAN$INSTRUCTION_BACKEND,
  },
  {
    id: "frontend",
    outputFile: "frontend-design.md",
    cites: ["Req", "DB", "BE"],
    defines: "FE",
    instructionKey: I18nKey.DEEP_PLAN$INSTRUCTION_FRONTEND,
  },
  {
    id: "tasks",
    outputFile: "tasks.md",
    cites: ["Req", "DB", "BE", "FE"],
    defines: null,
    instructionKey: I18nKey.DEEP_PLAN$INSTRUCTION_TASKS,
  },
  {
    id: "implementation",
    outputFile: null,
    cites: ["Req", "DB", "BE", "FE"],
    defines: null,
    instructionKey: I18nKey.DEEP_PLAN$INSTRUCTION_IMPLEMENTATION,
  },
] as const;

export const DEEP_PLAN_PHASE_IDS: readonly DeepPlanPhaseId[] =
  DEEP_PLAN_PHASES.map((phase) => phase.id);

/** The label a phase's sections carry, used by the validator to resolve citations. */
export const DEEP_PLAN_LABEL_TO_PHASE: Readonly<
  Record<DeepPlanLabel, DeepPlanPhaseId>
> = {
  Req: "requirements",
  DB: "database",
  BE: "backend",
  FE: "frontend",
};

export const getDeepPlanPhase = (id: DeepPlanPhaseId): DeepPlanPhase =>
  DEEP_PLAN_PHASES[DEEP_PLAN_PHASE_IDS.indexOf(id)];

/**
 * Maps a file the planner wrote to the phase that owns it, by basename, so the
 * reference validator has the documents to check at a checkpoint. Returns
 * `null` for any path that is not a phase output (e.g. `PLAN.md`).
 */
export function matchDeepPlanDocumentFile(
  path: string | null | undefined,
): DeepPlanPhaseId | null {
  if (!path) return null;
  const normalized = path.replace(/\\/g, "/").toUpperCase();
  const basename = normalized.slice(normalized.lastIndexOf("/") + 1);
  const phase = DEEP_PLAN_PHASES.find(
    (candidate) =>
      candidate.outputFile !== null &&
      candidate.outputFile.toUpperCase() === basename,
  );
  return phase?.id ?? null;
}

/** The phases a document is allowed to cite, in order. */
export const upstreamPhasesOf = (id: DeepPlanPhaseId): DeepPlanPhaseId[] =>
  DEEP_PLAN_PHASE_IDS.slice(0, DEEP_PLAN_PHASE_IDS.indexOf(id));

export const nextDeepPlanPhase = (
  id: DeepPlanPhaseId,
): DeepPlanPhaseId | null => {
  const index = DEEP_PLAN_PHASE_IDS.indexOf(id);
  return DEEP_PLAN_PHASE_IDS[index + 1] ?? null;
};
