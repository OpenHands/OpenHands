export const LOOPS_PATH = "/loops";
export const LOOPS_API_PATH = "/api/loops";
export const LOOPS_TRIGGERS_API_PATH = "/api/loops/triggers";
export const SESSION_API_KEY_HEADER = "X-Session-API-Key";

export const LOOP_TRIGGER_TYPES = [
  "scheduled",
  "on_commit",
  "on_pr",
  "manual",
] as const;

export const LOOP_SCHEDULE_TYPES = ["cron", "interval"] as const;

export const LOOP_RUN_STATUSES = [
  "pending",
  "running",
  "passed",
  "failed",
  "aborted",
  "awaiting_input",
] as const;

export const TRIGGER_EVENT_STATUSES = ["fired", "skipped", "error"] as const;

export const BUILTIN_LOOP_NAMES = [
  "commit-loop",
  "visual-regression",
  "perf-loop",
  "manual-loop",
  "doc-loop",
] as const;

export function loopRunPath(runId: string): string {
  return `${LOOPS_PATH}/runs/${encodeURIComponent(runId)}`;
}

export function loopRunIdFromPath(currentPath: string): string | null {
  const prefix = `${LOOPS_PATH}/runs/`;
  if (!currentPath.startsWith(prefix)) {
    return null;
  }
  const rest = currentPath.slice(prefix.length);
  if (!rest || rest.includes("/")) {
    return null;
  }
  return decodeURIComponent(rest);
}
