import type { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import { stripRedundantTitlePrefix } from "#/components/conversation-events/chat/event-content-helpers/get-acp-tool-call-content";

/**
 * Extract the command string from an ACP execute tool call event.
 * Prefers raw_input.command, then non-empty string raw_input, then stripped title.
 */
export function extractACPExecuteCommand(event: ACPToolCallEvent): string {
  const rawInput = event.raw_input;
  if (
    rawInput &&
    typeof rawInput === "object" &&
    "command" in rawInput &&
    typeof (rawInput as { command: unknown }).command === "string"
  ) {
    return (rawInput as { command: string }).command;
  }
  if (typeof rawInput === "string" && rawInput.trim()) {
    return rawInput.trim();
  }
  return stripRedundantTitlePrefix(event) || event.title || "";
}

/**
 * Extract output string from an ACP execute tool call event.
 * Checks content blocks (text or content), then raw_output.
 * If the call failed and no output was provided, returns a fallback error string.
 * Returns null if there is no output to display.
 */
export function extractACPExecuteOutput(
  event: ACPToolCallEvent,
): string | null {
  if (Array.isArray(event.content)) {
    const textParts: string[] = [];
    for (const block of event.content) {
      if (
        block.type === "text" &&
        typeof block.text === "string" &&
        block.text.trim()
      ) {
        textParts.push(block.text);
      } else if (block.type === "content") {
        const inner = block.content as Record<string, unknown> | null;
        if (inner && typeof inner === "object") {
          if (typeof inner.text === "string" && inner.text.trim()) {
            textParts.push(inner.text);
          } else if (
            inner.resource &&
            typeof inner.resource === "object" &&
            typeof (inner.resource as { text?: unknown }).text === "string"
          ) {
            const resText = (inner.resource as { text: string }).text;
            if (resText.trim()) {
              textParts.push(resText);
            }
          }
        }
      }
    }
    if (textParts.length > 0) {
      return textParts.join("\n").trim();
    }
  }

  if (typeof event.raw_output === "string") {
    const trimmed = event.raw_output.trim();
    if (trimmed) return trimmed;
  } else if (event.raw_output !== null && event.raw_output !== undefined) {
    try {
      const stringified = JSON.stringify(event.raw_output, null, 2);
      if (stringified && stringified !== "{}" && stringified !== "null") {
        return stringified;
      }
    } catch {
      const str = String(event.raw_output).trim();
      if (str) return str;
    }
  }

  if (event.is_error || event.status === "failed") {
    return "Command failed";
  }

  return null;
}
