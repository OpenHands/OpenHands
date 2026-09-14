import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import i18n from "#/i18n";
import { MAX_CONTENT_LENGTH } from "./shared";
import { I18nKey } from "#/i18n/declaration";
import { markdownFence, markdownInlineCode } from "#/utils/markdown-fence";

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

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;

type ACPDisplayBlock =
  | { type: "text"; text: string; language: string }
  | { type: "diff"; path: string; oldText: string | null; newText: string };

// A text block that is exactly one fenced code block (claude-agent-acp
// markdown-escapes file reads and shell output this way). Captures the fence,
// its language and the body so the body can be re-fenced after truncation
// without leaving an unterminated fence behind.
const WHOLLY_FENCED_RE = /^(`{3,})([^`\n]*)\n([\s\S]*?)\n?\1[ \t]*$/;

const toTextBlock = (text: string): ACPDisplayBlock | null => {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const fenced = WHOLLY_FENCED_RE.exec(trimmed);
  return fenced
    ? { type: "text", text: fenced[3], language: fenced[2].trim() }
    : { type: "text", text: trimmed, language: "" };
};

/**
 * Normalize ``event.content`` — the client-facing view of an ACP tool call —
 * into the blocks the card can display. The SDK persists blocks as snake_case
 * (``old_text``) while the ACP wire uses camelCase (``oldText``); both are
 * accepted. ``terminal`` blocks and non-text media carry nothing renderable
 * here and are skipped.
 */
const getACPDisplayBlocks = (event: ACPToolCallEvent): ACPDisplayBlock[] => {
  const blocks: ACPDisplayBlock[] = [];
  for (const block of event.content ?? []) {
    if (block.type === "diff") {
      const path = block.path;
      const newText = block.new_text ?? block.newText;
      const oldText = block.old_text ?? block.oldText ?? null;
      if (
        typeof path === "string" &&
        typeof newText === "string" &&
        (oldText === null || typeof oldText === "string")
      ) {
        blocks.push({ type: "diff", path, oldText, newText });
      }
    } else if (block.type === "content") {
      const inner = asRecord(block.content);
      let text: unknown;
      if (inner?.type === "text") {
        text = inner.text;
      } else if (inner?.type === "resource") {
        text = asRecord(inner.resource)?.text;
      }
      const textBlock = typeof text === "string" ? toTextBlock(text) : null;
      if (textBlock) blocks.push(textBlock);
    }
  }
  return blocks;
};

const diffLines = (text: string | null, prefix: string): string[] =>
  text
    ? text
        .replace(/\n$/, "")
        .split("\n")
        .map((line) => prefix + line)
    : [];

const formatDiffBlock = (
  block: Extract<ACPDisplayBlock, { type: "diff" }>,
): string => {
  const body = [
    ...diffLines(block.oldText, "- "),
    ...diffLines(block.newText, "+ "),
  ].join("\n");
  return `${markdownInlineCode(block.path)}\n${markdownFence(truncate(body), "diff")}`;
};

/**
 * Build the markdown-flavored body for an ACP tool call card. Mirrors the
 * shape of ``getTerminalObservationContent`` (``Command:`` + ``Output:``
 * fenced blocks) so the rendered card lines up with regular OpenHands
 * observations.
 *
 * The body prefers ``content``, the ACP field meant for display: ``diff``
 * blocks render as a diff under the file path, and text blocks as the output.
 * ``raw_output`` is provider-defined — absent for Gemini CLI, the model-facing
 * tool result for Claude Code — so it is only used when ``content`` has
 * nothing displayable, with the same "(no output)" fallback copy used by the
 * bash observation renderer.
 *
 * For ``tool_kind === "execute"`` we surface ``raw_input.command`` as the
 * command line; for others we fall back to a pretty-printed JSON dump of
 * the input, unless a diff already shows what the call changed.
 */
export const getACPToolCallContent = (event: ACPToolCallEvent): string => {
  const toolKind = event.tool_kind;
  const rawInput = event.raw_input;
  const rawOutput = event.raw_output;
  const isError = event.is_error;

  const blocks = getACPDisplayBlocks(event);
  const diffBlocks = blocks.filter((block) => block.type === "diff");
  const textBlocks = blocks.filter((block) => block.type === "text");

  const sections: string[] = [];

  // Input block — command for execute, JSON dump otherwise.
  if (
    toolKind === "execute" &&
    rawInput &&
    typeof rawInput === "object" &&
    "command" in rawInput &&
    typeof (rawInput as { command: unknown }).command === "string"
  ) {
    const { command } = rawInput as { command: string };
    sections.push(`Command: \`${command}\``);
  } else if (
    diffBlocks.length === 0 &&
    rawInput !== null &&
    rawInput !== undefined &&
    rawInput !== ""
  ) {
    const inputStr = stringifyPayload(rawInput);
    if (inputStr.trim()) {
      sections.push(`Input:\n${markdownFence(inputStr, "json")}`);
    }
  }

  sections.push(...diffBlocks.map(formatDiffBlock));

  const outputLabel = isError ? "**Error:**" : "Output:";
  if (textBlocks.length > 0) {
    const outputs = textBlocks.map((block) =>
      markdownFence(truncate(block.text), block.language),
    );
    sections.push(`${outputLabel}\n${outputs.join("\n\n")}`);
  } else {
    // Output block — matches the bash observation layout exactly. A diff
    // already shows what the call did, so raw_output is skipped alongside
    // one — except on failure, where it may be the only place the error lives.
    const outputStr = truncate(stringifyPayload(rawOutput).trim());
    if (diffBlocks.length === 0) {
      const outputBody =
        outputStr || i18n.t(I18nKey.OBSERVATION$COMMAND_NO_OUTPUT);
      sections.push(`${outputLabel}\n${markdownFence(outputBody)}`);
    } else if (isError && outputStr) {
      sections.push(`${outputLabel}\n${markdownFence(outputStr)}`);
    }
  }

  return sections.join("\n\n");
};
