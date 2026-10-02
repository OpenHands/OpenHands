/**
 * Deep Planning mode (#17819): a gated chain of documents where every section
 * cites the upstream sections it derives from, so the final implementation is
 * traceable back to a requirement.
 *
 * This module is the single source of truth for the phase order, the reference
 * vocabulary and the per-phase prompt. The reference *validation* lives in
 * `deep-plan-reference.ts` so it stays pure and unit-testable.
 */

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
  /** Phase-specific suffix appended to the planner's system prompt. */
  instruction: string;
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
    instruction: [
      "PHASE: Requirements analysis.",
      "Interview the user to establish scope, actors, constraints and success",
      "criteria. Ask focused questions; do not write files yet. When the picture is",
      "complete, summarize it and tell the user to confirm the phase to continue.",
    ].join("\n"),
  },
  {
    id: "requirements",
    outputFile: "requirements.md",
    cites: [],
    defines: "Req",
    instruction: [
      "PHASE: Requirements.",
      "Write `requirements.md` with numbered, SMART requirements. Every requirement",
      "section heading carries its number so it can be cited as `[Req X.X]`",
      "downstream. Include measurable non-functional requirements, roles and a",
      "version history. Chunk long documents (max 3 pages per write).",
    ].join("\n"),
  },
  {
    id: "database",
    outputFile: "database-design.md",
    cites: ["Req"],
    defines: "DB",
    instruction: [
      "PHASE: Database design.",
      "Write `database-design.md`. Every section cites the requirements it satisfies",
      "as `[Req X.X]`, and the section number itself is citable as `[DB X.X]`.",
      "Chunk if there are more than 15 tables.",
    ].join("\n"),
  },
  {
    id: "backend",
    outputFile: "backend-design.md",
    cites: ["Req", "DB"],
    defines: "BE",
    instruction: [
      "PHASE: Backend design.",
      "Write `backend-design.md`. Every section cites `[Req X.X] [DB X.X]`, and the",
      "section number itself is citable as `[BE X.X]`. Chunk if there are more than",
      "8 API modules.",
    ].join("\n"),
  },
  {
    id: "frontend",
    outputFile: "frontend-design.md",
    cites: ["Req", "DB", "BE"],
    defines: "FE",
    instruction: [
      "PHASE: Frontend design.",
      "Write `frontend-design.md`. Every section cites `[Req X.X] [DB X.X] [BE X.X]`,",
      "and the section number itself is citable as `[FE X.X]`. Always chunk: max 3",
      "pages per write.",
    ].join("\n"),
  },
  {
    id: "tasks",
    outputFile: "tasks.md",
    cites: ["Req", "DB", "BE", "FE"],
    defines: null,
    instruction: [
      "PHASE: Tasks.",
      "Write `tasks.md` as an ordered checklist. Every task cites `[Req X.X] [DB X.X]",
      "[BE X.X] [FE X.X]` and carries a priority, an estimate, and mandatory testing",
      "sub-tasks: Implement, Write unit tests, Write integration tests, Run tests &",
      "verify. Never mark a task complete while its tests fail.",
    ].join("\n"),
  },
  {
    id: "implementation",
    outputFile: null,
    cites: ["Req", "DB", "BE", "FE"],
    defines: null,
    instruction: [
      "PHASE: Implementation.",
      "Before implementing a task, read every section the task cites across all",
      "upstream documents — not just the task line. Work one task at a time and only",
      "tick it off once its tests pass.",
    ].join("\n"),
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
