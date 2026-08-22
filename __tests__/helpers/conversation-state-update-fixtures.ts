import { OpenHandsEvent } from "#/types/agent-server/core";

/**
 * A ConversationStateUpdateEvent reporting `execution_status` directly
 * (the `key: "execution_status"` variant). `timestamp` defaults to "now" as
 * an ISO string for tests that don't care about ordering; pass an explicit one
 * for tests that reconcile events by timestamp. The format matters even for
 * the default: the store compares timestamps with `localeCompare`, so an
 * epoch-millisecond string ("1770...") sorts *before* every ISO date rather
 * than after them — a "now" default that silently lands at the front.
 */
export const makeExecutionStatusUpdate = (
  id: string,
  status: string,
  timestamp: string = new Date().toISOString(),
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
  timestamp: string = new Date().toISOString(),
): OpenHandsEvent =>
  ({
    id,
    timestamp,
    source: "environment",
    kind: "ConversationStateUpdateEvent",
    key: "full_state",
    value: { execution_status: status },
  }) as unknown as OpenHandsEvent;
