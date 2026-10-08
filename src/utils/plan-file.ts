import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  DEEP_PLAN_PHASE_IDS,
  getDeepPlanPhase,
  type DeepPlanPhaseId,
} from "#/utils/deep-plan";

export const PLAN_RELATIVE_PATH = ".agents_tmp/PLAN.md";
export const PLANNING_SYSTEM_PROMPT_FILENAME = "system_prompt_planning.j2";
export const PLANNING_FILE_EDITOR_TOOL_NAME = "planning_file_editor";
export const LOCAL_PLANNER_PARENT_TAG_KEY = "plannerparent";
/** Tags a Deep Planning planner with the phase whose document it writes. */
export const DEEP_PLAN_PHASE_TAG_KEY = "plannerphase";
/** Directory the planner and the phase documents live under. */
export const AGENTS_TMP_DIR = ".agents_tmp";

const PLAN_FILENAME_UPPER = "PLAN.MD";

// Mirrors the SDK planning preset's `format_plan_structure()` output
// (software-agent-sdk openhands-tools/openhands/tools/preset/planning.py,
// `PLAN_STRUCTURE`). The planning agent loads `system_prompt_planning.j2` with
// this injected as `{{plan_structure}}`; keep it byte-identical to that SDK
// output so the local planner matches the canonical planning agent.
export const PLAN_STRUCTURE_TEXT = [
  "The plan must follow this structure exactly:",
  "",
  "1. OBJECTIVE",
  "   * Summarize the goal of the plan in one or two sentences.",
  "   * Restate the problem in clear operational terms.",
  "",
  "2. CONTEXT SUMMARY",
  "   * Briefly describe the relevant system components, files, or data involved.",
  "   * Mention any dependencies or constraints (technical, organizational, or external).",
  "",
  "3. APPROACH OVERVIEW",
  "   * Outline the chosen approach at a high level.",
  "   * Mention why it was selected (short rationale) if alternatives were considered.",
  "",
  "4. IMPLEMENTATION STEPS",
  "   * Provide a step-by-step plan for execution.",
  "   * Each step should include:",
  "     - a **goal** (what this step accomplishes),",
  "     - a **method** (how to do it, briefly),",
  "     - and optionally a **reference** (file, module, or function impacted).",
  "",
  "5. TESTING AND VALIDATION",
  "   * Describe how the implementation can be verified or validated.",
  "   * This section should describe what success looks like — expected outputs, behaviors, or conditions.",
].join("\n");

// System-prompt suffix for the local planning agent (mirrors the OpenHands
// app-server's PLANNING_AGENT_INSTRUCTION). The planner's directive + boundaries
// live in the system prompt; the planning conversation is created idle, so the
// user types the first message themselves and nothing is injected into the chat.
export const PLANNING_AGENT_INSTRUCTION = [
  "<IMPORTANT_PLANNING_BOUNDARIES>",
  "You are a Planning Agent that can ONLY create plans - you CANNOT execute code or make changes.",
  "",
  "Create or update the plan for the current task in the configured PLAN.md file.",
  "",
  "After you finalize the plan in PLAN.md:",
  '- Do NOT ask "Ready to proceed?" or offer to execute the plan',
  "- Do NOT attempt to run any implementation commands",
  "- Instead, tell the user they can click the **Build** button below the plan preview to switch to the code agent and execute the plan.",
  "",
  "Your role ends when the plan is finalized. Implementation is handled by the code agent.",
  "</IMPORTANT_PLANNING_BOUNDARIES>",
].join("\n");

export function buildPlanPath(workingDir: string): string {
  const normalized = workingDir.replace(/\/+$/, "");
  return `${normalized}/${PLAN_RELATIVE_PATH}`;
}

/**
 * The `plan_path` a Deep Planning planner is pinned to. The planning tool only
 * edits its `plan_path`, so each phase's planner is pointed at that phase's own
 * document — that is what lets the chain actually be produced. Pure-conversation
 * phases (`analysis`, `implementation`) have no output file and keep `PLAN.md`.
 */
export function buildPhasePlanPath(
  workingDir: string,
  phase: DeepPlanPhaseId,
): string {
  const normalized = workingDir.replace(/\/+$/, "");
  const outputFile = getDeepPlanPhase(phase).outputFile;
  return `${normalized}/${AGENTS_TMP_DIR}/${outputFile ?? "PLAN.md"}`;
}

export function isPlanFilePath(path: string | null | undefined): boolean {
  if (!path) return false;
  const normalized = path.replace(/\\/g, "/").toUpperCase();
  return (
    normalized === PLAN_FILENAME_UPPER ||
    normalized.endsWith(`/${PLAN_FILENAME_UPPER}`)
  );
}

/**
 * Whether `conversation` is the local planner helper for `parentConversationId`
 * — identity comes from the `plannerparent` tag `createLocalPlanningConversation`
 * stamps on creation, never from list position (`sub_conversation_ids` is the
 * generic, untyped child list).
 */
export function isPlannerConversationOf(
  conversation: Pick<AppConversation, "tags"> | null | undefined,
  parentConversationId: string,
): boolean {
  return (
    conversation?.tags?.[LOCAL_PLANNER_PARENT_TAG_KEY] === parentConversationId
  );
}

/** Finds the planner helper among a conversation's fetched sub-conversations. */
export function findPlannerConversationId(
  subConversations: (AppConversation | null)[] | null | undefined,
  parentConversationId: string | null | undefined,
): string | null {
  if (!parentConversationId) return null;
  return (
    subConversations?.find((sub) =>
      isPlannerConversationOf(sub, parentConversationId),
    )?.id ?? null
  );
}

/**
 * The Deep Planning phase a planner was created for, or `null` for a plain
 * `plan`-mode planner (which carries no phase tag). Lets the socket layer tell
 * a deep-plan planner pinned to `requirements.md` from one pinned to
 * `database-design.md` — they share the parent and the `plannerparent` tag.
 */
export function plannerPhaseOf(
  conversation: Pick<AppConversation, "tags"> | null | undefined,
): DeepPlanPhaseId | null {
  const phase = conversation?.tags?.[DEEP_PLAN_PHASE_TAG_KEY];
  return phase && (DEEP_PLAN_PHASE_IDS as readonly string[]).includes(phase)
    ? (phase as DeepPlanPhaseId)
    : null;
}

/**
 * Finds the planner helper for a specific Deep Planning phase. Falls back to
 * the plain planner when no phase is given, so `plan` mode is unchanged.
 */
export function findPhasePlannerConversationId(
  subConversations: (AppConversation | null)[] | null | undefined,
  parentConversationId: string | null | undefined,
  phase: DeepPlanPhaseId | null,
): string | null {
  if (!parentConversationId) return null;
  const planners = (subConversations ?? []).filter(
    (sub): sub is AppConversation =>
      sub !== null && isPlannerConversationOf(sub, parentConversationId),
  );
  if (phase) {
    return planners.find((sub) => plannerPhaseOf(sub) === phase)?.id ?? null;
  }
  // Plain planner: one without a phase tag.
  return planners.find((sub) => plannerPhaseOf(sub) === null)?.id ?? null;
}

/**
 * Whether a stored planner id may stand in for `parentConversationId`'s
 * planner for `phase` — the check every store-id fallback must pass.
 *
 * The store id is a single unscoped value that can still hold another
 * conversation's planner right after a navigation; unless we verify its owner
 * and phase, a fallback would route messages into a planner pinned to a
 * different document (or another conversation entirely).
 *
 * `recordedPhase` is the phase the store stamped when it wrote the id (the
 * store's `deepPlanPlannerPhase`; `null` for a plain planner). The fetched
 * `plannerparent` tag is authoritative when present; otherwise the id is trusted
 * only when its recorded phase matches and the list holds no planner for that
 * phase that would contradict it.
 */
export function isFallbackPlannerId(
  subConversations: (AppConversation | null)[] | null | undefined,
  parentConversationId: string | null | undefined,
  localPlanningConversationId: string | null | undefined,
  phase: DeepPlanPhaseId | null,
  recordedPhase: DeepPlanPhaseId | null,
): boolean {
  if (!parentConversationId || !localPlanningConversationId) return false;
  const planners = (subConversations ?? []).filter(
    (sub): sub is AppConversation =>
      sub !== null && isPlannerConversationOf(sub, parentConversationId),
  );
  const match = planners.find((sub) => sub.id === localPlanningConversationId);
  if (match) return plannerPhaseOf(match) === phase;
  // The tagged child is not in the fetched list. Trust the store id only when
  // it was recorded for this exact phase and no listed planner claims that
  // phase — a deep-plan phase must be proven, never assumed.
  return (
    recordedPhase === phase &&
    !planners.some((sub) => plannerPhaseOf(sub) === phase)
  );
}
