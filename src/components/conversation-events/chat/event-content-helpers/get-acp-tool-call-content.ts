import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import i18n from "#/i18n";
import { MAX_CONTENT_LENGTH } from "./shared";
import { I18nKey } from "#/i18n/declaration";

/**
 * Pick the translation key used for the ACP tool call title row. Mirrors
 * ACTION_MESSAGE$RUN / $EDIT / $READ etc.
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

// English verb prefixes ACP servers sometimes inline into the title.
// Claude Code emits ``"Read /Users/foo/bar"`` for a read tool — combined
// with the i18n template ``"Reading <cmd>{{title}}</cmd>"`` that lands as
// ``"Reading Read /Users/foo/bar"``. The redundant leading verb is the
// part we strip; the template's own verb (which is i18n'd) stays.
//
// Keyed by ``tool_kind`` so the strip is scoped to where double-verbing
// actually shows up. The english-only check is intentional — ACP servers
// are anglophone tools and emit english titles regardless of the user's
// canvas locale; matching translated verbs would mean every locale's
// strip list goes stale the moment a new server is added.
const REDUNDANT_TITLE_PREFIXES: Partial<Record<string, readonly string[]>> = {
  read: ["Read"],
  edit: ["Edit", "Write"],
  execute: ["Bash", "Run"],
  fetch: ["Fetch", "WebFetch"],
};

/**
 * Strip a leading verb from ``event.title`` that would duplicate the
 * verb baked into the i18n template (see ``REDUNDANT_TITLE_PREFIXES``).
 *
 * The match is anchored, case-sensitive, and requires the prefix to be
 * followed by whitespace so a token like ``"Reads"`` (an actual verb
 * elsewhere in the title) is left alone. If no prefix matches, the title
 * is returned verbatim.
 */
export const stripRedundantTitlePrefix = (event: ACPToolCallEvent): string => {
  const title = event.title;
  const tool_kind = event.tool_kind;
  if (!title || !tool_kind) return title;
  const prefixes = REDUNDANT_TITLE_PREFIXES[tool_kind];
  if (!prefixes) return title;
  for (const prefix of prefixes) {
    if (
      title.length > prefix.length &&
      title.startsWith(prefix) &&
      /\s/.test(title.charAt(prefix.length))
    ) {
      return title.slice(prefix.length).trimStart();
    }
  }
  return title;
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
  const outputBody = outputStr || i18n.t(I18nKey.OBSERVATION$COMMAND_NO_OUTPUT);
  output += `${outputLabel}\n\`\`\`\n${outputBody}\n\`\`\``;

  return output;
};
