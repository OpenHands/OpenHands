/**
 * Pure oracle for "did this exact bash command succeed in this conversation?".
 *
 * The agent's `terminal` tool runs commands in-process and reports them as
 * `ObservationEvent`s in the conversation's own event stream. It does NOT
 * publish to the agent-server's global `/api/bash/bash_events` stream — that
 * stream only carries commands the UI runs itself (git status, file listing),
 * so it cannot prove anything about the agent's command.
 *
 * Kept free of Playwright imports so it can be unit-tested in Vitest.
 */

const TERMINAL_OBSERVATION_KINDS = new Set([
  "TerminalObservation",
  "ExecuteBashObservation",
]);

/** Exit code the terminal tool reports while a command is still running. */
const STILL_RUNNING_EXIT_CODE = -1;

export interface ExpectedBashRun {
  /** The exact command string the agent was scripted to run. */
  command: string;
  /** A token that must appear in that command's own output. */
  token: string;
}

export type BashObservationVerdict =
  | { status: "success"; detail: string }
  | { status: "failed"; detail: string }
  | { status: "pending"; detail: string };

interface TerminalObservation {
  command: string;
  exitCode: number | null;
  output: string;
  isError: boolean;
  timedOut: boolean;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;
}

function contentText(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .map((item) => {
      const text = asRecord(item)?.text;
      return typeof text === "string" ? text : "";
    })
    .join("");
}

function toTerminalObservation(event: unknown): TerminalObservation | null {
  const observation = asRecord(asRecord(event)?.observation);
  if (!observation) return null;
  if (!TERMINAL_OBSERVATION_KINDS.has(String(observation.kind))) return null;

  return {
    command: typeof observation.command === "string" ? observation.command : "",
    exitCode:
      typeof observation.exit_code === "number" ? observation.exit_code : null,
    output: contentText(observation.content),
    isError: observation.is_error === true || observation.error === true,
    timedOut: observation.timeout === true,
  };
}

function failureReason(run: TerminalObservation, token: string): string | null {
  if (run.timedOut) return "timed out";
  if (run.isError) return "was reported as an error";
  if (run.exitCode !== 0) return `finished with exit code ${run.exitCode}`;
  if (!run.output.includes(token)) {
    return `output did not contain ${token}: ${JSON.stringify(run.output.slice(0, 200))}`;
  }
  return null;
}

function isFinished(run: TerminalObservation): boolean {
  return (
    run.timedOut ||
    run.isError ||
    (run.exitCode !== null && run.exitCode !== STILL_RUNNING_EXIT_CODE)
  );
}

/**
 * Evaluate a page of conversation events against the expected bash run.
 *
 * - `success`: an observation for exactly `expected.command` finished with
 *   exit code 0, no error/timeout flag, and `expected.token` in its output.
 * - `failed`: the command finished but none of its runs succeeded.
 * - `pending`: the command has not produced a finished observation yet.
 */
export function evaluateBashObservation(
  events: readonly unknown[],
  expected: ExpectedBashRun,
): BashObservationVerdict {
  const runs = events
    .map(toTerminalObservation)
    .filter(
      (run): run is TerminalObservation =>
        run !== null && run.command === expected.command,
    );

  if (runs.some((run) => failureReason(run, expected.token) === null)) {
    return { status: "success", detail: `${expected.command} succeeded` };
  }

  const finished = runs.filter(isFinished);
  if (finished.length > 0) {
    const reason = failureReason(finished[finished.length - 1], expected.token);
    return { status: "failed", detail: `${expected.command} ${reason}` };
  }

  return {
    status: "pending",
    detail:
      `${events.length} events, ${runs.length} unfinished ` +
      `observation(s) for ${expected.command}`,
  };
}
