import { ExecutionStatus } from "#/types/agent-server/core/base/common";

/**
 * Maps a conversation's execution status to a single emoji that visually
 * conveys the agent's state in the browser tab title.
 *
 *   ✅ green check — agent finished or is otherwise waiting on the user
 *   🟢 green circle — agent is actively running
 *   ⚪ gray circle — agent is paused/stopped
 *   🔴 red circle — agent is in an error state
 *
 * Returns null when the status is unknown so callers can omit the prefix.
 */
export function getAgentStateEmoji(
  status: ExecutionStatus | null | undefined,
): string | null {
  switch (status) {
    case ExecutionStatus.RUNNING:
      return "🟢";
    case ExecutionStatus.FINISHED:
    case ExecutionStatus.IDLE:
    case ExecutionStatus.WAITING_FOR_CONFIRMATION:
      return "✅";
    case ExecutionStatus.PAUSED:
      return "⚪";
    case ExecutionStatus.ERROR:
    case ExecutionStatus.STUCK:
      return "🔴";
    default:
      return null;
  }
}
