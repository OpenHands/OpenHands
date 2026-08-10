import { OpenHandsEvent } from "#/types/agent-server/core";

/**
 * A ConversationStateUpdateEvent reporting `execution_status` directly
 * (the `key: "execution_status"` variant). `timestamp` defaults to "now" for
 * tests that don't care about ordering; pass an explicit ISO string for tests
 * that reconcile events by timestamp.
 */
export const makeExecutionStatusUpdate = (
  id: string,
  status: string,
  timestamp: string = Date.now().toString(),
): OpenHandsEvent =>
  ({
    id,
    timestamp,
    source: "environment",
    kind: "ConversationStateUpdateEvent",
    key: "execution_status",
    value: status,
  }) as unknown as OpenHandsEvent;

/**
 * A ConversationStateUpdateEvent carrying a full state snapshot (the
 * `key: "full_state"` variant), as sent live while the agent is still
 * running. See `makeExecutionStatusUpdate` for the `timestamp` default.
 */
export const makeFullStateSnapshot = (
  id: string,
  status: string,
  timestamp: string = Date.now().toString(),
): OpenHandsEvent =>
  ({
    id,
    timestamp,
    source: "environment",
    kind: "ConversationStateUpdateEvent",
    key: "full_state",
    value: { execution_status: status },
  }) as unknown as OpenHandsEvent;
