import { ToolCallID } from "../base/common";
import { BaseEvent } from "../base/event";

/**
 * Tool kinds emitted by ACP agents. Matches ACP's ``ToolKind`` enum,
 * with ``"other"`` as the catch-all fallback.
 */
export type ACPToolKind = "execute" | "edit" | "read" | "fetch" | "other";

/**
 * Status of an ACP tool call. The SDK persists two events per
 * ``tool_call_id``: an early ``started`` event (``pending`` / ``in_progress``)
 * and one terminal event (``completed`` / ``failed``) — the
 * action->observation pair. Non-terminal statuses render the card as
 * "running"; see ``getACPToolCallResult``.
 */
export type ACPToolCallStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed";

/**
 * An ACP content block as surfaced on ``ACPToolCallEvent.content``. ACP
 * allows a mix of text, image, resource and resource_link blocks; only the
 * shape needed for rendering is captured here.
 */
export interface ACPToolCallContentBlock {
  type: string;
  [key: string]: unknown;
}

/**
 * ACPToolCallEvent — surfaces a tool call executed by an ACP subprocess
 * (Claude Code, Codex, Gemini CLI, …) so the GUI can render it the same
 * way those tools' native UIs do.
 */
export interface ACPToolCallEvent extends BaseEvent {
  /**
   * Discriminator for the V1 event union.
   */
  kind: "ACPToolCallEvent";

  /**
   * ACP sub-agent is the event source; kept as ``"agent"`` in the SDK.
   */
  source: "agent";

  /**
   * Stable id assigned by the ACP server for this tool call. Multiple
   * events may share the same ``tool_call_id`` as the call progresses.
   */
  tool_call_id: ToolCallID;

  /**
   * Human-readable title, e.g. the command being executed or the path
   * being edited.
   */
  title: string;

  /**
   * Current lifecycle status of the call. ``null`` is tolerated for
   * backwards compatibility with events produced before the field was
   * required.
   */
  status: ACPToolCallStatus | null;

  /**
   * Category of tool: execute (shell), edit, read, fetch or other.
   */
  tool_kind: ACPToolKind | null;

  /**
   * Raw input payload as reported by the ACP server. Shape depends on
   * ``tool_kind`` — e.g. ``{ command }`` for execute, ``{ path, content }``
   * for edit.
   */
  raw_input: unknown;

  /**
   * Raw output payload — typically a string for execute / read tools,
   * or a structured value for fetch / other.
   */
  raw_output: unknown;

  /**
   * Optional ACP content blocks associated with the tool call output.
   */
  content: ACPToolCallContentBlock[] | null;

  /**
   * True if the ACP server reported this tool call as an error, even
   * when ``status`` is ``completed``.
   */
  is_error: boolean;
}
