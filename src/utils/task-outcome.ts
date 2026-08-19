import {
  TASK_OUTCOME_STATUSES,
  type AutomationRunMetadata,
  type TaskOutcome,
  type TaskOutcomeBlocker,
  type TaskOutcomeStatus,
} from "#/types/automation";

export const FINISH_TOOL_RESPONSE_METADATA_KEY = "finish_tool_response";

const TASK_OUTCOME_STATUS_SET = new Set<TaskOutcomeStatus>(
  TASK_OUTCOME_STATUSES,
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeStatus(value: unknown): TaskOutcomeStatus {
  return typeof value === "string" &&
    TASK_OUTCOME_STATUS_SET.has(value as TaskOutcomeStatus)
    ? (value as TaskOutcomeStatus)
    : "unknown";
}

function normalizeBlockers(value: unknown): TaskOutcomeBlocker[] {
  if (!Array.isArray(value)) return [];

  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const message = stringOrNull(item.message);
    if (!message) return [];

    const type = stringOrNull(item.type);
    return [
      {
        ...(type && { type }),
        message,
        ...(typeof item.recoverable === "boolean" && {
          recoverable: item.recoverable,
        }),
      },
    ];
  });
}

export function normalizeTaskOutcome(value: unknown): TaskOutcome | null {
  if (!isRecord(value)) return null;

  const outcomeSummary = stringOrNull(value.outcome_summary);
  const status = normalizeStatus(value.status);
  const blockers = normalizeBlockers(value.blockers);
  const needsUserAction = value.needs_user_action === true;

  if (
    !outcomeSummary &&
    status === "unknown" &&
    blockers.length === 0 &&
    !needsUserAction
  ) {
    return null;
  }

  return {
    status,
    outcome_summary: outcomeSummary,
    blockers,
    confidence: numberOrNull(value.confidence),
    needs_user_action: needsUserAction,
  };
}

export function getTaskOutcomeFromRunMetadata(
  metadata: AutomationRunMetadata | null | undefined,
): TaskOutcome | null {
  return normalizeTaskOutcome(metadata?.[FINISH_TOOL_RESPONSE_METADATA_KEY]);
}
