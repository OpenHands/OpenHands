import { BaseEvent } from "../base/event";

/**
 * Transient LLM token delta emitted while the agent's LLM streams a response
 * (when the LLM has ``stream=True``).
 *
 * These events are NOT part of the durable conversation record: the agent
 * server publishes them straight to its WebSocket subscribers and never
 * persists them. A client that reconnects mid-stream receives the final
 * ``MessageEvent`` from history but none of the deltas that produced it — they
 * are purely a live-rendering affordance. Consecutive deltas are merged in the
 * event store, and reconciled into (then superseded by) the final
 * ``MessageEvent`` / ``FinishAction`` for the turn.
 */
export interface StreamingDeltaEvent extends BaseEvent {
  kind: "StreamingDeltaEvent";
  source: "agent";
  /** Incremental chunk of the assistant's user-facing message text. */
  content: string | null;
  /** Incremental chunk of the LLM's reasoning/thinking text. */
  reasoning_content: string | null;
}
