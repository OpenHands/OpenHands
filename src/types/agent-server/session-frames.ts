import { OpenHandsEvent } from "./core";

/**
 * Frames of `/sockets/session/{conversation_id}` (agent-server
 * `session_protocol.py`). The envelope is deliberately not an `Event`: the
 * durable record rides inside it untouched, and protocol fields live on the
 * envelope. The URL is the protocol version — there is no handshake.
 *
 * Unknown frame types must be ignored: the envelope is allowed to grow.
 */

/** Sent once, before any replay, describing the range about to be sent. */
export interface SyncFrame {
  type: "sync";
  from_seq?: number | null;
  through_seq?: number | null;
}

/** One persisted event, after it is safely on disk. `seq` is its log index. */
export interface DurableFrame {
  type: "durable";
  seq: number;
  event: OpenHandsEvent;
}

/** An event that is published but never persisted, so it carries no `seq`. */
export interface TransientFrame {
  type: "transient";
  event: OpenHandsEvent;
}

/**
 * A stream is opening. `item_id` is the id the durable event will be built
 * with, so the slot it opens is retired by `event.id === item_id`.
 */
export interface ItemStartedFrame {
  type: "item_started";
  item_id: string;
  attempt?: number;
  anchor_seq?: number | null;
}

/** One masked increment of a stream. `order` is monotonic per (item, attempt). */
export interface DeltaFrame {
  type: "delta";
  item_id: string;
  attempt?: number;
  order: number;
  kind?: "text" | "reasoning";
  content: string;
  chunk_id?: string | null;
  choice_index?: number | null;
}

/** A stream ended without producing a durable event. */
export interface ItemAbortedFrame {
  type: "item_aborted";
  item_id: string;
  attempt?: number;
  reason: string;
}

/** A problem with this socket. Distinct from a durable conversation error. */
export interface SessionErrorFrame {
  type: "error";
  code: string;
  detail: string;
}

export type SessionFrame =
  | SyncFrame
  | DurableFrame
  | TransientFrame
  | ItemStartedFrame
  | DeltaFrame
  | ItemAbortedFrame
  | SessionErrorFrame;

const FRAME_TYPES = new Set<SessionFrame["type"]>([
  "sync",
  "durable",
  "transient",
  "item_started",
  "delta",
  "item_aborted",
  "error",
]);

/**
 * Narrow a decoded socket message to a known frame, or `null` for anything
 * this client does not recognize.
 */
export const asSessionFrame = (value: unknown): SessionFrame | null => {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const { type } = value as { type?: unknown };
  if (
    typeof type !== "string" ||
    !FRAME_TYPES.has(type as SessionFrame["type"])
  ) {
    return null;
  }
  return value as SessionFrame;
};
