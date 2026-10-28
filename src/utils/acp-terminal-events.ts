import { useCommandStore } from "#/stores/command-store";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";

// A text block that is exactly one fenced code block (claude-agent-acp
// markdown-escapes shell output this way). The terminal renders the body
// without the fence. Mirrors `getACPToolCallContent`.
const WHOLLY_FENCED_RE = /^(`{3,})([^`\n]*)\n([\s\S]*?)\n?\1[ \t]*$/;

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;

/**
 * True once the call can no longer change — mirrors `getACPToolCallResult`
 * (`failed` / `is_error` are errors, `completed` is success, everything else
 * is still running).
 */
const isTerminalACPToolCall = (event: ACPToolCallEvent): boolean =>
  event.is_error === true ||
  event.status === "failed" ||
  event.status === "completed";

/** The command of an ACP execute call, when its raw input carries one. */
const getExecuteCommand = (event: ACPToolCallEvent): string | null => {
  if (event.tool_kind !== "execute") return null;
  const command = asRecord(event.raw_input)?.command;
  return typeof command === "string" && command ? command : null;
};

/**
 * Plain-text output of an ACP tool call for the terminal. Mirrors the output
 * half of `getACPToolCallContent`: text carried in `content` blocks wins
 * (the display field — Gemini CLI puts all output there), falling back to
 * the provider-defined `raw_output` (a string for execute, structured JSON
 * otherwise). Unlike the chat card there is no truncation or markdown
 * fencing — the terminal renders raw text, like ExecuteBashObservation.
 */
export const getACPTerminalOutput = (event: ACPToolCallEvent): string => {
  const texts: string[] = [];
  for (const block of event.content ?? []) {
    if (block.type !== "content") continue;
    const inner = asRecord(block.content);
    const text =
      inner?.type === "text"
        ? inner.text
        : inner?.type === "resource"
          ? asRecord(inner.resource)?.text
          : undefined;
    if (typeof text !== "string" || !text.trim()) continue;
    const trimmed = text.trim();
    const fenced = WHOLLY_FENCED_RE.exec(trimmed);
    texts.push(fenced ? fenced[3] : trimmed);
  }
  if (texts.length > 0) return texts.join("\n");

  const rawOutput = event.raw_output;
  if (rawOutput === null || rawOutput === undefined) return "";
  if (typeof rawOutput === "string") return rawOutput.trim();
  try {
    return JSON.stringify(rawOutput, null, 2);
  } catch {
    return String(rawOutput);
  }
};

/**
 * Mirror an ACP execute tool call into the Terminal tab (#18252), matching
 * what the built-in terminal gets from ExecuteBashAction / Observation pairs.
 *
 * The SDK persists several events per `tool_call_id` (an early
 * started/pending one and one terminal event at least), so each half is
 * appended at most once, keyed by the id recorded on the command row: the
 * command on the first event seen, the output on the terminal event. An
 * `is_error` flag counts as terminal even when `status` still says
 * otherwise, mirroring `getACPToolCallResult`. Non-execute kinds (read,
 * edit, fetch, …) add nothing — the terminal shows shell activity only.
 * Replayed history is additionally filtered by event id in the WebSocket
 * handlers (#1656), which keeps reconnects from double-appending.
 */
export const handleACPToolCallInTerminal = (event: ACPToolCallEvent): void => {
  const command = getExecuteCommand(event);
  if (command === null) return;

  const { commands } = useCommandStore.getState();
  if (
    !commands.some(
      (c) => c.type === "input" && c.toolCallId === event.tool_call_id,
    )
  ) {
    useCommandStore.getState().appendInput(command, event.tool_call_id);
  }

  if (!isTerminalACPToolCall(event)) return;

  const { commands: afterCommandAppend } = useCommandStore.getState();
  if (
    afterCommandAppend.some(
      (c) => c.type === "output" && c.toolCallId === event.tool_call_id,
    )
  ) {
    return;
  }

  const output = getACPTerminalOutput(event);
  if (output) {
    useCommandStore.getState().appendOutput(output, event.tool_call_id);
  }
};
