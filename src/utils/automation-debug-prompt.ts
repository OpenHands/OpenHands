export interface AutomationDebugPromptInput {
  /** Human-readable automation name, if known. */
  automationName?: string;
  /** The automation's instructions (what it was set up to do). */
  automationPrompt?: string | null;
  /** Run-level error string; the durable fallback when stderr is unavailable. */
  errorDetail?: string | null;
  /** The run's captured stderr (the traceback shown in the Error tab). */
  stderr?: string | null;
  /** The failed run's id, included as a reference. */
  runId: string;
}

// Keep the seeded first message bounded. A traceback's most useful part is the
// tail (the actual error), so when the captured output is huge we keep the end.
const MAX_ERROR_CHARS = 4000;

function keepTail(text: string, max: number): string {
  if (text.length <= max) return text;
  return `…(truncated)…\n${text.slice(text.length - max)}`;
}

/**
 * Build the first message for a "Debug with OpenHands" conversation started
 * from a failed automation run. This is an instruction to the agent (not
 * user-facing UI copy), so it is intentionally in English and not localized.
 */
export function buildAutomationDebugPrompt({
  automationName,
  automationPrompt,
  errorDetail,
  stderr,
  runId,
}: AutomationDebugPromptInput): string {
  // Prefer the live stderr the user is looking at; fall back to the run-level
  // error_detail when the sandbox is gone and stderr couldn't be fetched.
  const rawError = stderr?.trim() || errorDetail?.trim() || "";
  const errorSection = rawError
    ? keepTail(rawError, MAX_ERROR_CHARS)
    : "No error output was captured for this run.";

  const trimmedName = automationName?.trim();
  const intro = trimmedName
    ? `The scheduled automation "${trimmedName}" failed during a run.`
    : "A scheduled automation failed during a run.";

  const lines: string[] = [
    `${intro} Please investigate the error and fix the root cause.`,
  ];

  const trimmedPrompt = automationPrompt?.trim();
  if (trimmedPrompt) {
    lines.push("", "What the automation was set up to do:", trimmedPrompt);
  }

  lines.push(
    "",
    `Error output from the failed run (run ${runId}):`,
    errorSection,
    "",
    "Please diagnose why it failed and propose or implement a fix.",
  );

  return lines.join("\n");
}
