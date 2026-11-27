import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isExecuteBashObservationEvent,
  isExecuteBashActionEvent,
  isDisplayableErrorEvent,
} from "#/types/agent-server/type-guards";

export interface DetectedSandboxOOM {
  /** Id of the observation or error event that signalled OOM. */
  eventId: string;
  timestamp: string;
  command?: string;
  detail?: string;
}

const OOM_TEXT_PATTERNS = [
  /\bOOMKilled\b/i,
  /\bOut of memory\b/i,
  /\bcgroup.*out of memory\b/i,
  /\bKilled\b/,
];

export function isOOMText(text: string): boolean {
  return OOM_TEXT_PATTERNS.some((pattern) => pattern.test(text));
}

/**
 * Scans conversation events for Out-Of-Memory (OOM) termination markers.
 * Linux kernel / container cgroup OOM kills processes with SIGKILL (128 + 9 = 137).
 */
export function detectSandboxOOM(
  events: OpenHandsEvent[],
): DetectedSandboxOOM[] {
  const actionCommands = new Map<string, string>();
  for (const event of events) {
    if (isExecuteBashActionEvent(event)) {
      actionCommands.set(event.id, event.action.command);
    }
  }

  const oomEvents: DetectedSandboxOOM[] = [];

  for (const event of events) {
    if (isExecuteBashObservationEvent(event)) {
      const exitCode = event.observation.exit_code;
      const text = event.observation.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");

      if (exitCode === 137 || isOOMText(text)) {
        oomEvents.push({
          eventId: event.id,
          timestamp: event.timestamp,
          command: actionCommands.get(event.action_id),
          detail: text.slice(0, 300),
        });
      }
    } else if (isDisplayableErrorEvent(event)) {
      const errorEvent = event as {
        id: string;
        timestamp: string;
        detail?: string;
        code?: string;
      };
      if (
        errorEvent.code?.includes("137") ||
        errorEvent.code?.toLowerCase().includes("oom") ||
        (errorEvent.detail && isOOMText(errorEvent.detail))
      ) {
        oomEvents.push({
          eventId: errorEvent.id,
          timestamp: errorEvent.timestamp,
          detail: errorEvent.detail,
        });
      }
    }
  }

  return oomEvents;
}
