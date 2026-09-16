import { BaseEvent } from "../base/event";

/**
 * One open streaming slot, rendered in place of the message that will replace
 * it. Client-minted, never persisted, and not an agent-server event: `id` is
 * the `item_id` of the `item_started` frame that opened it, which is also the
 * id the durable event is built with — so the slot is retired by a single
 * equality test rather than by comparing its text to the message.
 */
export interface StreamingDeltaEvent extends BaseEvent {
  kind: "StreamingDeltaEvent";
  source: "agent";
  content: string | null;
  reasoning_content: string | null;
  /** Attempt currently rendered; a higher one supersedes this text. */
  attempt?: number;
}
