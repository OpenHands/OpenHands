import type {
  ActionEvent,
  OpenHandsEvent,
  ObservationEvent,
} from "#/types/agent-server/core";
import type { TaskAction } from "#/types/agent-server/core/base/action";
import type { TaskObservation } from "#/types/agent-server/core/base/observation";
import {
  isACPToolCallEvent,
  isActionEvent,
  isAgentErrorEvent,
  isConversationStateUpdateEvent,
  isGoalConversationStateUpdateEvent,
  isHookExecutionEvent,
  isMessageEvent,
  isObservationEvent,
  isStreamingDeltaEvent,
  isSwitchLLMObservationEvent,
} from "#/types/agent-server/type-guards";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import { parseMessageFromEvent } from "#/components/conversation-events/chat/event-content-helpers/parse-message-from-event";
import { getActionContent } from "#/components/conversation-events/chat/event-content-helpers/get-action-content";
import { getObservationContent } from "#/components/conversation-events/chat/event-content-helpers/get-observation-content";
import {
  getACPToolCallContent,
  stripRedundantTitlePrefix,
} from "#/components/conversation-events/chat/event-content-helpers/get-acp-tool-call-content";
import { groupEvents } from "#/components/conversation-events/chat/group-events";
import {
  getActionThoughtText,
  getReasoningContent,
  splitInlineThink,
} from "#/components/conversation-events/chat/event-thought-helpers";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";

export type TranscriptExportFormat = "markdown" | "html";

export interface TranscriptExportOptions {
  includeToolDetails: boolean;
  includeTimestamps: boolean;
  title?: string | null;
  model?: string | null;
}

type TranscriptEntry =
  | {
      kind: "message";
      author: "user" | "assistant";
      content: string;
      timestamp: string;
    }
  | {
      kind: "tool";
      summary: string;
      details: string;
      timestamp: string;
    }
  | {
      kind: "error";
      content: string;
      timestamp: string;
    }
  | {
      kind: "note";
      summary: string;
      content: string;
      timestamp: string;
    };

const SAFE_ACTION_DETAIL_KINDS = new Set([
  "BrowserClickAction",
  "BrowserCloseTabAction",
  "BrowserGetContentAction",
  "BrowserGetStateAction",
  "BrowserGoBackAction",
  "BrowserListTabsAction",
  "BrowserNavigateAction",
  "BrowserScrollAction",
  "BrowserSwitchTabAction",
  "BrowserTypeAction",
  "ExecuteBashAction",
  "FileEditorAction",
  "GlobAction",
  "GrepAction",
  "InvokeSkillAction",
  "MCPToolAction",
  "StrReplaceEditorAction",
  "TaskAction",
  "TaskTrackerAction",
  "TerminalAction",
  "ThinkAction",
]);

const SAFE_OBSERVATION_DETAIL_KINDS = new Set([
  "BrowserObservation",
  "CanvasUIObservation",
  "ExecuteBashObservation",
  "FileEditorObservation",
  "GlobObservation",
  "GrepObservation",
  "InvokeSkillObservation",
  "MCPToolObservation",
  "StrReplaceEditorObservation",
  "SwitchLLMObservation",
  "TaskTrackerObservation",
  "TaskObservation",
  "TerminalObservation",
]);

const cleanInlineText = (value: string): string =>
  value
    .replace(/[\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const truncate = (value: string, maxLength = 100): string =>
  value.length > maxLength ? `${value.slice(0, maxLength)}…` : value;

const formatTimestamp = (timestamp: string): string => {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? timestamp : date.toISOString();
};

const escapeHtml = (value: string): string =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

/**
 * GitHub Markdown permits raw HTML, so dynamic conversation text needs the
 * same inert representation as the chat's sanitized Markdown renderer.
 */
const sanitizeMarkdownText = (value: string): string =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replace(
      /(!?)\[([^\]]*)\]\(\s*((?:javascript|data|vbscript):[^)]*)\)/gi,
      "$1\\[$2\\]($3)",
    )
    .replace(
      /^(\s{0,3})\[([^\]]+)\]:(\s*<?(?:javascript|data|vbscript):)/gim,
      "$1\\[$2\\]:$3",
    );

const safeTitle = (title?: string | null): string =>
  cleanInlineText(title || i18n.t(I18nKey.TRANSCRIPT_EXPORT$DEFAULT_TITLE)) ||
  i18n.t(I18nKey.TRANSCRIPT_EXPORT$DEFAULT_TITLE);

const translatePlain = (
  key: I18nKey,
  values?: Record<string, unknown>,
): string =>
  cleanInlineText(i18n.t(key, values).replace(/<\/?(?:cmd|path)>/g, ""));

const isServerFallbackSummary = (summary: string): boolean =>
  /^[a-z][a-z0-9_]*\s*:\s*[[{]/i.test(summary);

const humanizeEventKind = (kind: string): string =>
  kind
    .replace(/(?:Action|Observation)$/, "")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .trim();

const getActionSummary = (event: ActionEvent): string => {
  const summary = cleanInlineText(event.summary || "");
  if (summary && !isServerFallbackSummary(summary)) return summary;
  return (
    cleanInlineText(event.tool_name) || humanizeEventKind(event.action.kind)
  );
};

const getObservationSummary = (
  event: ObservationEvent,
  correspondingAction?: ActionEvent,
): string => {
  if (correspondingAction) return getActionSummary(correspondingAction);
  return (
    cleanInlineText(event.tool_name) ||
    humanizeEventKind(event.observation.kind)
  );
};

const getTextContent = (
  content: Array<{ type: string; text?: string }>,
): string =>
  content
    .filter(
      (item): item is { type: "text"; text: string } =>
        item.type === "text" && typeof item.text === "string",
    )
    .map((item) => item.text)
    .join("\n");

const getTaskActionDetails = (action: TaskAction): string =>
  [
    `${i18n.t(I18nKey.TASK$SUBAGENT)}: ${action.subagent_type}`,
    `${i18n.t(I18nKey.TASK$QUERY)}:\n${action.prompt}`,
  ].join("\n\n");

const getTaskObservationDetails = (
  observation: TaskObservation,
  correspondingAction?: ActionEvent,
): string => {
  const taskAction =
    correspondingAction?.action.kind === "TaskAction"
      ? correspondingAction.action
      : undefined;
  return [
    `${i18n.t(I18nKey.TASK$SUBAGENT)}: ${observation.subagent}`,
    `${i18n.t(I18nKey.TASK$TASK_ID)}: ${observation.task_id}`,
    taskAction ? `${i18n.t(I18nKey.TASK$QUERY)}:\n${taskAction.prompt}` : "",
    `${i18n.t(I18nKey.TASK$RESULT)}:\n${getTextContent(observation.content)}`,
  ]
    .filter(Boolean)
    .join("\n\n");
};

const getSafeActionDetails = (event: ActionEvent): string => {
  if (event.action.kind === "TaskAction") {
    return getTaskActionDetails(event.action);
  }
  return SAFE_ACTION_DETAIL_KINDS.has(event.action.kind)
    ? getActionContent(event)
    : "";
};

const getSafeObservationDetails = (
  event: ObservationEvent,
  correspondingAction?: ActionEvent,
): string => {
  if (event.observation.kind === "TaskObservation") {
    return getTaskObservationDetails(event.observation, correspondingAction);
  }
  return SAFE_OBSERVATION_DETAIL_KINDS.has(event.observation.kind)
    ? getObservationContent(event)
    : "";
};

const getHookDetails = (event: OpenHandsEvent): string => {
  if (!isHookExecutionEvent(event)) return "";
  return [event.reason, event.error, event.stdout, event.stderr]
    .filter((value): value is string => Boolean(value?.trim()))
    .join("\n\n");
};

const getActionNarration = (event: ActionEvent): string =>
  [getReasoningContent(event), getActionThoughtText(event)]
    .map((content) => content.trim())
    .filter(Boolean)
    .join("\n\n");

const canRenderActionNarration = (event: ActionEvent): boolean =>
  !["FinishAction", "SwitchLLMAction", "ThinkAction"].includes(
    event.action.kind,
  );

const buildTranscriptEntries = (
  events: OpenHandsEvent[],
  includeToolDetails: boolean,
): TranscriptEntry[] => {
  const uiEvents = events.reduce<OpenHandsEvent[]>((current, event) => {
    try {
      return handleEventForUI(event, current);
    } catch {
      return current;
    }
  }, []);
  const actionsById = new Map(
    events.filter(isActionEvent).map((event) => [event.id, event]),
  );
  const entries: TranscriptEntry[] = [];
  const emittedNarrationActionIds = new Set<string>();
  const renderableEvents = uiEvents.filter(
    (event) =>
      (isSwitchLLMObservationEvent(event) && !event.observation.is_error) ||
      shouldRenderEvent(event),
  );
  const renderedItems = groupEvents(
    renderableEvents,
    Number.MAX_SAFE_INTEGER,
    events,
  );

  const addActionNarration = (action: ActionEvent) => {
    if (
      emittedNarrationActionIds.has(action.id) ||
      !canRenderActionNarration(action)
    ) {
      return;
    }
    emittedNarrationActionIds.add(action.id);
    const content = getActionNarration(action);
    if (content) {
      entries.push({
        kind: "message",
        author: "assistant",
        content,
        timestamp: action.timestamp,
      });
    }
  };

  for (const item of renderedItems) {
    if (item.kind === "group") continue;
    if (item.kind === "thought") {
      addActionNarration(item.action);
      continue;
    }

    const { event } = item;
    try {
      const narrationAction = isActionEvent(event)
        ? event
        : isObservationEvent(event)
          ? actionsById.get(event.action_id)
          : undefined;
      if (narrationAction) addActionNarration(narrationAction);

      if (isSwitchLLMObservationEvent(event) && !event.observation.is_error) {
        entries.push({
          kind: "note",
          summary: translatePlain(I18nKey.MODEL$SWITCHED_TO_PROFILE, {
            name: event.observation.profile_name,
          }),
          content: [
            event.observation.active_model
              ? `${i18n.t(I18nKey.TRANSCRIPT_EXPORT$MODEL)}: ${event.observation.active_model}`
              : "",
            event.observation.reason
              ? `${i18n.t(I18nKey.TRANSCRIPT_EXPORT$REASON)}: ${event.observation.reason}`
              : "",
          ]
            .filter(Boolean)
            .join("\n"),
          timestamp: event.timestamp,
        });
        continue;
      }

      if (isMessageEvent(event)) {
        const parsed = parseMessageFromEvent(event).trim();
        if (event.source === "agent") {
          const { reasoning, message } = splitInlineThink(parsed);
          [reasoning, message].filter(Boolean).forEach((content) => {
            entries.push({
              kind: "message",
              author: "assistant",
              content,
              timestamp: event.timestamp,
            });
          });
        } else if (parsed) {
          entries.push({
            kind: "message",
            author: "user",
            content: parsed,
            timestamp: event.timestamp,
          });
        }
        continue;
      }

      if (isStreamingDeltaEvent(event)) {
        const { reasoning: inlineThink, message } = splitInlineThink(
          event.content ?? "",
          { streaming: true },
        );
        const reasoningContent = [
          event.reasoning_content?.trim() || "",
          inlineThink,
        ]
          .filter(Boolean)
          .join("\n\n");
        if (reasoningContent) {
          entries.push({
            kind: "message",
            author: "assistant",
            content: reasoningContent,
            timestamp: event.timestamp,
          });
        }
        if (message.trim()) {
          entries.push({
            kind: "message",
            author: "assistant",
            content: message.trim(),
            timestamp: event.timestamp,
          });
        }
        continue;
      }

      if (isAgentErrorEvent(event)) {
        entries.push({
          kind: "error",
          content: event.error,
          timestamp: event.timestamp,
        });
        continue;
      }

      if (isActionEvent(event)) {
        if (event.action.kind === "FinishAction") {
          const content = event.action.message.trim();
          if (content) {
            entries.push({
              kind: "message",
              author: "assistant",
              content,
              timestamp: event.timestamp,
            });
          }
        } else {
          entries.push({
            kind: "tool",
            summary: getActionSummary(event),
            details: includeToolDetails ? getSafeActionDetails(event) : "",
            timestamp: event.timestamp,
          });
        }
        continue;
      }

      if (isObservationEvent(event)) {
        const correspondingAction = actionsById.get(event.action_id);
        entries.push({
          kind: "tool",
          summary: getObservationSummary(event, correspondingAction),
          details: includeToolDetails
            ? getSafeObservationDetails(event, correspondingAction)
            : "",
          timestamp: event.timestamp,
        });
        continue;
      }

      if (isACPToolCallEvent(event)) {
        entries.push({
          kind: "tool",
          summary:
            stripRedundantTitlePrefix(event) ||
            i18n.t(I18nKey.ACTION_MESSAGE$ACP_TOOL),
          details: includeToolDetails ? getACPToolCallContent(event) : "",
          timestamp: event.timestamp,
        });
        continue;
      }

      if (isHookExecutionEvent(event)) {
        entries.push({
          kind: "tool",
          summary: i18n.t(I18nKey.TRANSCRIPT_EXPORT$HOOK, {
            command: truncate(cleanInlineText(event.hook_command), 100),
          }),
          details: includeToolDetails ? getHookDetails(event) : "",
          timestamp: event.timestamp,
        });
        continue;
      }

      if (
        isConversationStateUpdateEvent(event) &&
        isGoalConversationStateUpdateEvent(event)
      ) {
        entries.push({
          kind: "note",
          summary: `${i18n.t(I18nKey.GOAL$PREFIX)} ${i18n.t(
            {
              running: I18nKey.GOAL$STATUS_RUNNING,
              complete: I18nKey.GOAL$STATUS_COMPLETE,
              capped: I18nKey.GOAL$STATUS_CAPPED,
              interrupted: I18nKey.GOAL$STATUS_INTERRUPTED,
            }[event.value.status],
          )}`,
          content: [event.value.objective, event.value.verdict?.missing || ""]
            .filter(Boolean)
            .join("\n\n"),
          timestamp: event.timestamp,
        });
      }
    } catch {
      // Runtime event payloads may be older than the local TypeScript schema.
      // Skip malformed entries instead of aborting an otherwise valid export.
    }
  }

  return entries;
};

const markdownTimestamp = (
  entry: TranscriptEntry,
  options: TranscriptExportOptions,
): string =>
  options.includeTimestamps
    ? `<sub>${escapeHtml(formatTimestamp(entry.timestamp))}</sub>\n\n`
    : "";

const markdownFence = (content: string): string => {
  const longestRun = Math.max(
    0,
    ...Array.from(content.matchAll(/`+/g), (match) => match[0].length),
  );
  const fence = "`".repeat(Math.max(3, longestRun + 1));
  return `${fence}text\n${content}\n${fence}`;
};

export const eventsToMarkdown = (
  events: OpenHandsEvent[],
  options: TranscriptExportOptions,
): string => {
  const lines = [`# ${sanitizeMarkdownText(safeTitle(options.title))}`, ""];

  if (options.model) {
    lines.push(
      `**${i18n.t(I18nKey.TRANSCRIPT_EXPORT$MODEL)}:** ${sanitizeMarkdownText(cleanInlineText(options.model))}`,
      "",
    );
  }

  for (const entry of buildTranscriptEntries(
    events,
    options.includeToolDetails,
  )) {
    const timestamp = markdownTimestamp(entry, options);
    if (entry.kind === "message") {
      const author =
        entry.author === "user"
          ? i18n.t(I18nKey.TRANSCRIPT_EXPORT$USER)
          : i18n.t(I18nKey.CHAT_INTERFACE$ASSISTANT);
      lines.push(
        `## ${author}`,
        "",
        timestamp + sanitizeMarkdownText(entry.content),
        "",
      );
    } else if (entry.kind === "error") {
      const quoted = entry.content
        .split("\n")
        .map((line) => `> ${sanitizeMarkdownText(line)}`)
        .join("\n");
      lines.push(
        `## ${i18n.t(I18nKey.COMMON$ERROR)}`,
        "",
        timestamp + quoted,
        "",
      );
    } else if (entry.kind === "note") {
      lines.push(
        `> **${escapeHtml(entry.summary)}**`,
        "",
        timestamp + sanitizeMarkdownText(entry.content),
        "",
      );
    } else if (options.includeToolDetails && entry.details) {
      lines.push(
        "<details>",
        `<summary><strong>${escapeHtml(i18n.t(I18nKey.TRANSCRIPT_EXPORT$TOOL))}:</strong> ${escapeHtml(entry.summary)}</summary>`,
        "",
        timestamp + markdownFence(entry.details),
        "",
        "</details>",
        "",
      );
    } else {
      const timestampMarkup = timestamp ? `<br>${timestamp.trimEnd()}` : "";
      lines.push(
        `<p><strong>${escapeHtml(i18n.t(I18nKey.TRANSCRIPT_EXPORT$TOOL))}:</strong> ${escapeHtml(entry.summary)}${timestampMarkup}</p>`,
        "",
      );
    }
  }

  return `${lines.join("\n").trim()}\n`;
};

const htmlTimestamp = (
  entry: TranscriptEntry,
  options: TranscriptExportOptions,
): string => {
  if (!options.includeTimestamps) return "";
  const timestamp = formatTimestamp(entry.timestamp);
  return `<time datetime="${escapeHtml(timestamp)}">${escapeHtml(timestamp)}</time>`;
};

export const eventsToHtml = (
  events: OpenHandsEvent[],
  options: TranscriptExportOptions,
): string => {
  const title = safeTitle(options.title);
  const body = buildTranscriptEntries(events, options.includeToolDetails)
    .map((entry) => {
      const timestamp = htmlTimestamp(entry, options);
      if (entry.kind === "message") {
        const author =
          entry.author === "user"
            ? i18n.t(I18nKey.TRANSCRIPT_EXPORT$USER)
            : i18n.t(I18nKey.CHAT_INTERFACE$ASSISTANT);
        return `<section class="message ${entry.author.toLowerCase()}">
  <header><h2>${escapeHtml(author)}</h2>${timestamp}</header>
  <div class="content">${escapeHtml(entry.content)}</div>
</section>`;
      }
      if (entry.kind === "error") {
        return `<section class="message error">
  <header><h2>${escapeHtml(i18n.t(I18nKey.COMMON$ERROR))}</h2>${timestamp}</header>
  <div class="content">${escapeHtml(entry.content)}</div>
</section>`;
      }
      if (entry.kind === "note") {
        return `<aside class="note">
  <header><strong>${escapeHtml(entry.summary)}</strong>${timestamp}</header>
  ${entry.content ? `<div class="content">${escapeHtml(entry.content)}</div>` : ""}
</aside>`;
      }
      const details =
        options.includeToolDetails && entry.details
          ? `<details>
  <summary><strong>${escapeHtml(i18n.t(I18nKey.TRANSCRIPT_EXPORT$TOOL))}:</strong> ${escapeHtml(entry.summary)}${timestamp}</summary>
  <pre>${escapeHtml(entry.details)}</pre>
</details>`
          : `<div class="tool-summary"><strong>${escapeHtml(i18n.t(I18nKey.TRANSCRIPT_EXPORT$TOOL))}:</strong> ${escapeHtml(entry.summary)}${timestamp}</div>`;
      return details;
    })
    .join("\n");

  const model = options.model
    ? `<p class="model"><strong>${escapeHtml(i18n.t(I18nKey.TRANSCRIPT_EXPORT$MODEL))}:</strong> ${escapeHtml(cleanInlineText(options.model))}</p>`
    : "";

  return `<!doctype html>
<html lang="${escapeHtml(i18n.resolvedLanguage || i18n.language || "en")}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
  <title>${escapeHtml(title)}</title>
  <style>
    :root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #e5e7eb; line-height: 1.55; }
    main { box-sizing: border-box; width: min(860px, 100%); margin: 0 auto; padding: 48px 24px 80px; }
    h1 { margin: 0; font-size: 2rem; }
    h2 { margin: 0; font-size: 1rem; }
    .model { margin: 8px 0 32px; color: #9ca3af; }
    .message, details, .tool-summary, .note { margin: 16px 0; border: 1px solid #374151; border-radius: 10px; padding: 16px; background: #1f2937; }
    .user { border-left: 4px solid #60a5fa; }
    .assistant { border-left: 4px solid #34d399; }
    .error { border-left: 4px solid #f87171; }
    .note { border-left: 4px solid #a78bfa; }
    header, summary, .tool-summary { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
    summary { cursor: pointer; }
    time { flex: none; color: #9ca3af; font-size: .75rem; font-weight: 400; }
    .content { margin-top: 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
    pre { margin: 14px 0 0; padding: 14px; overflow-x: auto; border-radius: 7px; background: #111827; color: #d1d5db; white-space: pre-wrap; overflow-wrap: anywhere; }
    @media (prefers-color-scheme: light) {
      body { background: #f9fafb; color: #111827; }
      .message, details, .tool-summary, .note { border-color: #d1d5db; background: #fff; }
      pre { background: #f3f4f6; color: #1f2937; }
      .model, time { color: #6b7280; }
    }
  </style>
</head>
<body>
  <main>
    <h1>${escapeHtml(title)}</h1>
    ${model}
    ${body}
  </main>
</body>
</html>
`;
};
