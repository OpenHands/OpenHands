import { ACPToolCallEvent } from "#/types/v1/core/events/acp-tool-call-event";
import i18n from "#/i18n";
import { MAX_CONTENT_LENGTH } from "./shared";

/**
 * Pick the translation key used for the ACP tool call title row. Mirrors
 * ACTION_MESSAGE$RUN / $EDIT / $READ etc. but with an ``ACP · `` prefix so
 * the card is visually distinguishable from regular OpenHands actions.
 */
export const getACPToolCallTitleKey = (event: ACPToolCallEvent): string => {
  switch (event.tool_kind) {
    case "execute":
      return "ACTION_MESSAGE$ACP_RUN";
    case "edit":
      return "ACTION_MESSAGE$ACP_EDIT";
    case "read":
      return "ACTION_MESSAGE$ACP_READ";
    case "fetch":
      return "ACTION_MESSAGE$ACP_FETCH";
    default:
      return "ACTION_MESSAGE$ACP_TOOL";
  }
};

/**
 * Stringify an arbitrary raw_input / raw_output payload for markdown
 * rendering. Strings pass through; objects are pretty-printed JSON.
 */
const stringifyPayload = (value: unknown): string => {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const truncate = (content: string): string =>
  content.length > MAX_CONTENT_LENGTH
    ? `${content.slice(0, MAX_CONTENT_LENGTH)}...`
    : content;

/**
 * Build the markdown-flavored body for an ACP tool call card. Mirrors the
 * shape of ``getTerminalObservationContent`` (``Command:`` + ``Output:``
 * fenced blocks) so the rendered card lines up with regular OpenHands
 * observations.
 *
 * For ``tool_kind === "execute"`` we surface ``raw_input.command`` as the
 * command line; for others we fall back to a pretty-printed JSON dump of
 * the input. Output is always dumped as a fenced block, with the same
 * "(no output)" fallback copy used by the bash observation renderer.
 */
export const getACPToolCallContent = (event: ACPToolCallEvent): string => {
  const toolKind = event.tool_kind;
  const rawInput = event.raw_input;
  const rawOutput = event.raw_output;
  const isError = event.is_error;

  let output = "";

  // Input block — command for execute, JSON dump otherwise.
  if (
    toolKind === "execute" &&
    rawInput &&
    typeof rawInput === "object" &&
    "command" in rawInput &&
    typeof (rawInput as { command: unknown }).command === "string"
  ) {
    const { command } = rawInput as { command: string };
    output += `Command: \`${command}\`\n\n`;
  } else if (rawInput !== null && rawInput !== undefined && rawInput !== "") {
    const inputStr = stringifyPayload(rawInput);
    if (inputStr.trim()) {
      output += `Input:\n\`\`\`json\n${inputStr}\n\`\`\`\n\n`;
    }
  }

  // Output block — matches the bash observation layout exactly.
  const outputStr = truncate(stringifyPayload(rawOutput).trim());
  const outputLabel = isError ? "**Error:**" : "Output:";
  const outputBody = outputStr || i18n.t("OBSERVATION$COMMAND_NO_OUTPUT");
  output += `${outputLabel}\n\`\`\`\n${outputBody}\n\`\`\``;

  return output;
};
