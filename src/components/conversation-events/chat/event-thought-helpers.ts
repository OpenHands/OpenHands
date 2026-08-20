import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import { ThinkingBlock } from "#/types/agent-server/core/base/event";
import {
  isActionEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";

/**
 * Returns the displayable thought text of an `ActionEvent`, or an empty
 * string if the event has no usable thought content.
 *
 * Mirrors the logic used by `ThoughtEventMessage` so callers stay in sync
 * with what gets rendered.
 */
export const getActionThoughtText = (action: ActionEvent): string =>
  action.thought
    .filter((t) => t.type === "text")
    .map((t) => t.text)
    .join("\n");

/**
 * Extracts extended thinking / reasoning content from an `ActionEvent`.
 *
 * Prefers `reasoning_content` (a plain string produced by many reasoning
 * models). Falls back to the text from `thinking_blocks` (Anthropic
 * extended thinking). Returns an empty string when neither is available.
 */
export const getReasoningContent = (action: ActionEvent): string => {
  if (action.reasoning_content) {
    return action.reasoning_content;
  }

  if (action.thinking_blocks?.length) {
    return action.thinking_blocks
      .filter((b): b is ThinkingBlock => b.type === "thinking")
      .map((b) => b.thinking)
      .join("\n\n");
  }

  return "";
};

export const hasNonEmptyThought = (action: ActionEvent): boolean =>
  getActionThoughtText(action).trim().length > 0;

/**
 * Splits a leading `<think>…</think>` reasoning block out of assistant content
 * so it renders in the collapsible thinking section, not the message bubble.
 * Some models stream reasoning inline instead of via `reasoning_content`.
 *
 * Conservative to avoid mangling normal messages: only a `<think>` at the very
 * start is touched (later occurrences, e.g. quoted in docs, stay verbatim),
 * only the first block is peeled, and an unclosed leading `<think>` is reasoning
 * only while `streaming` — in a finalized message it's literal output.
 */
export const splitInlineThink = (
  content: string,
  options?: { streaming?: boolean },
): { reasoning: string; message: string } => {
  const OPEN = "<think>";
  const CLOSE = "</think>";

  // Only a <think> at the very start is reasoning.
  const leading = content.replace(/^\s+/, "");
  if (!leading.startsWith(OPEN)) {
    return { reasoning: "", message: content };
  }

  const afterOpen = leading.slice(OPEN.length);
  const close = afterOpen.indexOf(CLOSE);

  if (close === -1) {
    // Unclosed: reasoning-in-progress while streaming, else literal output.
    return options?.streaming
      ? { reasoning: afterOpen.trim(), message: "" }
      : { reasoning: "", message: content };
  }

  return {
    reasoning: afterOpen.slice(0, close).trim(),
    message: afterOpen.slice(close + CLOSE.length).trim(),
  };
};

/**
 * Find the `ActionEvent` whose thought should be rendered alongside the
 * given UI event. For an `ActionEvent` the thought belongs to itself; for
 * an `ObservationEvent` we look up the matching action in `allEvents`.
 *
 * `ThinkAction` is intentionally excluded because its thought IS the
 * action body and is rendered through a separate codepath.
 */
export const getThoughtSourceAction = (
  event: OpenHandsEvent,
  allEvents: OpenHandsEvent[],
): ActionEvent | null => {
  if (isActionEvent(event)) {
    if (event.action.kind === "ThinkAction") return null;
    return hasNonEmptyThought(event) ? event : null;
  }

  if (isObservationEvent(event)) {
    const action = allEvents.find(
      (e): e is ActionEvent => isActionEvent(e) && e.id === event.action_id,
    );
    if (!action) return null;
    if (action.action.kind === "ThinkAction") return null;
    return hasNonEmptyThought(action) ? action : null;
  }

  return null;
};
