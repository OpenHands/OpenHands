export const PLAN_RELATIVE_PATH = ".agents_tmp/PLAN.md";
export const PLANNING_SYSTEM_PROMPT_FILENAME = "system_prompt_planning.j2";
export const PLANNING_FILE_EDITOR_TOOL_NAME = "planning_file_editor";
export const LOCAL_PLANNER_PARENT_TAG_KEY = "plannerparent";

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

export function isPlanFilePath(path: string | null | undefined): boolean {
  if (!path) return false;
  const normalized = path.replace(/\\/g, "/").toUpperCase();
  return (
    normalized === PLAN_FILENAME_UPPER ||
    normalized.endsWith(`/${PLAN_FILENAME_UPPER}`)
  );
}
