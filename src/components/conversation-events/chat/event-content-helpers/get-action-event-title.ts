import { CANVAS_UI_CLIENT_ACTION_KIND } from "#/constants/canvas-ui";
import { LAUNCH_CHILD_CONVERSATION_ACTION_KIND } from "#/constants/child-conversation";
import type { ActionEvent } from "#/types/agent-server/core";

export type EventTitleDescriptor =
  | {
      readonly kind: "translation";
      readonly key: string;
      readonly values: Readonly<Record<string, unknown>>;
    }
  | {
      readonly kind: "text";
      readonly text: string;
    };

export const trimEventTitleText = (text: string, maxLength: number): string => {
  if (!text) return "";
  return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
};

const isServerFallbackSummary = (summary: string): boolean =>
  /^[a-z][a-z0-9_]*\s*:\s*[[{]/i.test(summary);

export const getActionSummaryTitle = (event: ActionEvent): string | null => {
  const summary = event.summary?.trim().replace(/\s+/g, " ") || "";
  return !summary || isServerFallbackSummary(summary) ? null : summary;
};

export const getActionEventTitleDescriptor = (
  event: ActionEvent,
): EventTitleDescriptor => {
  const summary = getActionSummaryTitle(event);
  if (summary) {
    return { kind: "text", text: summary };
  }

  const actionType = event.action.kind;

  switch (actionType) {
    case "ExecuteBashAction":
    case "TerminalAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$RUN",
        values: { command: trimEventTitleText(event.action.command, 80) },
      };
    case "FileEditorAction":
    case "StrReplaceEditorAction": {
      const key =
        event.action.command === "view"
          ? "ACTION_MESSAGE$READ"
          : event.action.command === "create"
            ? "ACTION_MESSAGE$WRITE"
            : "ACTION_MESSAGE$EDIT";
      return {
        kind: "translation",
        key,
        values: { path: event.action.path },
      };
    }
    case "MCPToolAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$CALL_TOOL_MCP",
        values: { mcp_tool_name: event.tool_name },
      };
    case "InvokeSkillAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$INVOKE_SKILL",
        values: { name: event.action.name },
      };
    case "TaskAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$TASK",
        values: { name: event.action.subagent_type },
      };
    case "ThinkAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$THINK",
        values: {},
      };
    case "FinishAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$FINISH",
        values: {},
      };
    case "TaskTrackerAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$TASK_TRACKING",
        values: {},
      };
    case "GrepAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$GREP",
        values: {
          pattern: event.action.pattern
            ? trimEventTitleText(event.action.pattern, 50)
            : "",
        },
      };
    case "GlobAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$GLOB",
        values: {
          pattern: event.action.pattern
            ? trimEventTitleText(event.action.pattern, 50)
            : "",
        },
      };
    case "BrowserNavigateAction":
    case "BrowserClickAction":
    case "BrowserTypeAction":
    case "BrowserGetStateAction":
    case "BrowserGetContentAction":
    case "BrowserScrollAction":
    case "BrowserGoBackAction":
    case "BrowserListTabsAction":
    case "BrowserSwitchTabAction":
    case "BrowserCloseTabAction":
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$BROWSE",
        values: {},
      };
    case "CanvasUIAction":
    case CANVAS_UI_CLIENT_ACTION_KIND:
      return { kind: "text", text: "CANVASUI" };
    case LAUNCH_CHILD_CONVERSATION_ACTION_KIND:
      return {
        kind: "translation",
        key: "ACTION_MESSAGE$LAUNCH_CHILD_CONVERSATION",
        values: {},
      };
    default:
      return {
        kind: "text",
        text: String(actionType).replace("Action", "").toUpperCase(),
      };
  }
};
