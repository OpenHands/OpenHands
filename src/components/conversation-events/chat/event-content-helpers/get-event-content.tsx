import { Trans } from "react-i18next";
import React from "react";
import {
  CANVAS_UI_CLIENT_ACTION_KIND,
  CANVAS_UI_CLIENT_TOOL_NAME,
} from "#/constants/canvas-ui";
import {
  OpenHandsEvent,
  ObservationEvent,
  ActionEvent,
} from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
  isACPToolCallEvent,
  isCanvasUIActionEvent,
} from "#/types/agent-server/type-guards";
import { MonoComponent } from "../../../features/chat/mono-component";
import { PathComponent } from "../../../features/chat/path-component";
import { getActionContent } from "./get-action-content";
import { getObservationContent } from "./get-observation-content";
import {
  getACPToolCallContent,
  getACPToolCallTitleKey,
  stripRedundantTitlePrefix,
} from "./get-acp-tool-call-content";
import { TaskTrackingObservationContent } from "../task-tracking/task-tracking-observation-content";
import { TaskTrackerObservation } from "#/types/agent-server/core/base/observation";
import { SkillReadyEvent, isSkillReadyEvent } from "./create-skill-ready-event";
import { resolveVisualizerBody } from "../../../features/chat/tool-visualizers/dispatcher";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";

const trimText = (text: string, maxLength: number): string => {
  if (!text) return "";
  return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
};

// Helper function to create title from translation key
const createTitleFromKey = (
  key: string,
  values: Record<string, unknown>,
): React.ReactNode => {
  if (!i18n.exists(key)) {
    return key;
  }

  return (
    <Trans
      ns="openhands"
      i18nKey={key}
      values={values}
      components={{
        path: <PathComponent />,
        cmd: <MonoComponent />,
      }}
    />
  );
};

/**
 * Detects the agent-server's default summary fallback, which has the shape
 * `{tool_name}: {json-args}` (see `_extract_summary` in
 * `openhands/sdk/agent/agent.py`). When the LLM omits a summary the server
 * dumps the raw arguments JSON, which renders as a huge unreadable blob in
 * the chat. Treat that case as "no summary" so the action-kind specific
 * title (e.g. "Editing <path>", "Running <cmd>") is used instead.
 */
const isServerFallbackSummary = (summary: string): boolean =>
  /^[a-z][a-z0-9_]*\s*:\s*[[{]/i.test(summary);

const getSummaryTitleForActionEvent = (
  event: ActionEvent,
): React.ReactNode | null => {
  const summary = event.summary?.trim().replace(/\s+/g, " ") || "";
  if (!summary || isServerFallbackSummary(summary)) {
    return null;
  }
  return summary;
};

const getNonEmptyString = (
  value: unknown,
  maxLength?: number,
): string | null => {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  return maxLength ? trimText(trimmed, maxLength) : trimmed;
};

const getVisionInspectActionTitle = (
  event: ActionEvent,
): React.ReactNode | null => {
  if (event.tool_name !== "inspect_image_with_vision") {
    return null;
  }

  const action = event.action as unknown as Record<string, unknown>;
  const profileName = getNonEmptyString(action.profile_name);

  return profileName
    ? `Describing image with ${profileName}`
    : "Describing image with auxiliary vision LLM";
};

const getVisionInspectObservationTitle = (
  event: ObservationEvent,
  correspondingAction?: ActionEvent,
): React.ReactNode | null => {
  const actionToolName = correspondingAction?.tool_name;
  const observation = event.observation as unknown as Record<string, unknown>;
  const observationToolName = getNonEmptyString(observation.tool_name);

  if (
    actionToolName !== "inspect_image_with_vision" &&
    observationToolName !== "inspect_image_with_vision" &&
    String(event.observation.kind) !== "VisionInspectObservation"
  ) {
    return null;
  }

  const action = (correspondingAction?.action ?? {}) as unknown as Record<
    string,
    unknown
  >;
  const label =
    getNonEmptyString(action.profile_name) ??
    getNonEmptyString(observation.profile_name) ??
    getNonEmptyString(observation.base_url, 80) ??
    getNonEmptyString(observation.endpoint, 80) ??
    getNonEmptyString(observation.model, 80);

  return label
    ? `Describing image with ${label}`
    : "Describing image with auxiliary vision LLM";
};

// Action Event Processing
const getActionEventTitle = (event: OpenHandsEvent): React.ReactNode => {
  // Early return if not an action event
  if (!isActionEvent(event)) {
    return "";
  }

  const visionInspectTitle = getVisionInspectActionTitle(event);
  if (visionInspectTitle) {
    return visionInspectTitle;
  }

  const summaryTitle = getSummaryTitleForActionEvent(event);
  if (summaryTitle) {
    return summaryTitle;
  }

  const actionType = event.action.kind;
  let actionKey = "";
  let actionValues: Record<string, unknown> = {};

  switch (actionType) {
    case "ExecuteBashAction":
    case "TerminalAction":
      actionKey = "ACTION_MESSAGE$RUN";
      actionValues = {
        command: trimText(event.action.command, 80),
      };
      break;
    case "FileEditorAction":
    case "StrReplaceEditorAction":
      if (event.action.command === "view") {
        actionKey = "ACTION_MESSAGE$READ";
      } else if (event.action.command === "create") {
        actionKey = "ACTION_MESSAGE$WRITE";
      } else {
        actionKey = "ACTION_MESSAGE$EDIT";
      }
      actionValues = {
        path: event.action.path,
      };
      break;
    case "MCPToolAction":
      actionKey = "ACTION_MESSAGE$CALL_TOOL_MCP";
      actionValues = {
        mcp_tool_name: event.tool_name,
      };
      break;
    case "InvokeSkillAction":
      actionKey = "ACTION_MESSAGE$INVOKE_SKILL";
      actionValues = {
        name: event.action.name,
      };
      break;
    case "TaskAction":
      actionKey = "ACTION_MESSAGE$TASK";
      actionValues = {
        name: event.action.subagent_type,
      };
      break;
    case "ThinkAction":
      actionKey = "ACTION_MESSAGE$THINK";
      break;
    case "FinishAction":
      actionKey = "ACTION_MESSAGE$FINISH";
      break;
    case "TaskTrackerAction":
      actionKey = "ACTION_MESSAGE$TASK_TRACKING";
      break;
    case "GrepAction":
      actionKey = "ACTION_MESSAGE$GREP";
      actionValues = {
        pattern:
          "pattern" in event.action && event.action.pattern
            ? trimText(String(event.action.pattern), 50)
            : "",
      };
      break;
    case "GlobAction":
      actionKey = "ACTION_MESSAGE$GLOB";
      actionValues = {
        pattern:
          "pattern" in event.action && event.action.pattern
            ? trimText(String(event.action.pattern), 50)
            : "",
      };
      break;
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
      actionKey = "ACTION_MESSAGE$BROWSE";
      break;
    case "CanvasUIAction":
    case CANVAS_UI_CLIENT_ACTION_KIND:
      return "CANVASUI";
    default:
      // For unknown actions, use the type name
      return String(actionType).replace("Action", "").toUpperCase();
  }

  if (actionKey) {
    return createTitleFromKey(actionKey, actionValues);
  }

  return actionType;
};

// Observation Event Processing
const getObservationEventTitle = (
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): React.ReactNode => {
  // Early return if not an observation event
  if (!isObservationEvent(event)) {
    return "";
  }

  const visionInspectObservationTitle = getVisionInspectObservationTitle(
    event,
    correspondingAction,
  );
  if (visionInspectObservationTitle) {
    return visionInspectObservationTitle;
  }

  if (correspondingAction) {
    const visionInspectTitle = getVisionInspectActionTitle(correspondingAction);
    if (visionInspectTitle) {
      return visionInspectTitle;
    }

    const summaryTitle = getSummaryTitleForActionEvent(correspondingAction);
    if (summaryTitle) {
      return summaryTitle;
    }
  }

  const observationType = event.observation.kind;
  let observationKey = "";
  let observationValues: Record<string, unknown> = {};

  switch (observationType) {
    case "ExecuteBashObservation":
    case "TerminalObservation":
      observationKey = "OBSERVATION_MESSAGE$RUN";
      observationValues = {
        command: event.observation.command
          ? trimText(event.observation.command, 80)
          : "",
      };
      break;
    case "FileEditorObservation":
    case "StrReplaceEditorObservation":
      if (event.observation.command === "view") {
        observationKey = "OBSERVATION_MESSAGE$READ";
      } else {
        observationKey = "OBSERVATION_MESSAGE$EDIT";
      }
      observationValues = {
        path: event.observation.path || "",
      };
      break;
    case "MCPToolObservation":
      observationKey = "OBSERVATION_MESSAGE$MCP";
      observationValues = {
        mcp_tool_name: event.observation.tool_name,
      };
      break;
    case "InvokeSkillObservation":
      observationKey = "OBSERVATION_MESSAGE$INVOKE_SKILL";
      observationValues = {
        name: event.observation.skill_name,
      };
      break;
    case "TaskObservation":
      observationKey = "OBSERVATION_MESSAGE$TASK";
      observationValues = {
        name: event.observation.subagent,
      };
      break;
    case "CanvasUIObservation":
      observationKey = "OBSERVATION_MESSAGE$CANVAS_UI";
      break;
    case "ClientToolObservation":
      if (event.tool_name === CANVAS_UI_CLIENT_TOOL_NAME) {
        observationKey = "OBSERVATION_MESSAGE$CANVAS_UI";
        break;
      }
      return observationType.replace("Observation", "").toUpperCase();
    case "SwitchLLMObservation":
      observationKey = event.observation.is_error
        ? "MODEL$SWITCH_FAILED"
        : "MODEL$SWITCHED_TO_PROFILE";
      observationValues = {
        name: event.observation.profile_name,
      };
      break;
    case "BrowserObservation":
      observationKey = "OBSERVATION_MESSAGE$BROWSE";
      break;
    case "TaskTrackerObservation": {
      const { command } = event.observation;
      if (command === "plan") {
        observationKey = "OBSERVATION_MESSAGE$TASK_TRACKING_PLAN";
      } else {
        // command === "view"
        observationKey = "OBSERVATION_MESSAGE$TASK_TRACKING_VIEW";
      }
      break;
    }
    case "ThinkObservation":
      observationKey = "OBSERVATION_MESSAGE$THINK";
      break;
    case "GlobObservation":
      observationKey = "OBSERVATION_MESSAGE$GLOB";
      observationValues = {
        pattern: event.observation.pattern
          ? trimText(event.observation.pattern, 50)
          : "",
      };
      break;
    case "GrepObservation":
      observationKey = "OBSERVATION_MESSAGE$GREP";
      observationValues = {
        pattern: event.observation.pattern
          ? trimText(event.observation.pattern, 50)
          : "",
      };
      break;
    default:
      // For unknown observations, use the type name
      return observationType.replace("Observation", "").toUpperCase();
  }

  if (observationKey) {
    return createTitleFromKey(observationKey, observationValues);
  }

  return observationType;
};

const getCanvasUIClientObservationContent = (
  event: ObservationEvent,
  correspondingAction?: ActionEvent,
): string | null => {
  if (
    event.observation.kind !== "ClientToolObservation" ||
    event.tool_name !== CANVAS_UI_CLIENT_TOOL_NAME ||
    !correspondingAction ||
    !isCanvasUIActionEvent(correspondingAction)
  ) {
    return null;
  }

  return `UI command '${correspondingAction.action.command}' dispatched to the Agent Canvas frontend.`;
};

export const getEventContent = (
  event: OpenHandsEvent | SkillReadyEvent,
  correspondingAction?: ActionEvent,
) => {
  let title: React.ReactNode = "";
  let details: string | React.ReactNode = "";

  // Handle Skill Ready events first
  if (isSkillReadyEvent(event)) {
    // Use translation key if available, otherwise use "SKILL READY"
    const skillReadyKey = "OBSERVATION_MESSAGE$SKILL_READY";
    if (i18n.exists(skillReadyKey)) {
      title = createTitleFromKey(skillReadyKey, {});
    } else {
      title = "Skill Ready";
    }
    details = event._skillReadyContent;
  } else if (isActionEvent(event)) {
    title = getActionEventTitle(event);
    // Per-tool React visualizer when one is registered; markdown otherwise.
    details = resolveVisualizerBody(event) ?? getActionContent(event);
  } else if (isObservationEvent(event)) {
    title = getObservationEventTitle(event, correspondingAction);

    // For TaskTrackerObservation, use React component instead of markdown
    if (event.observation.kind === "TaskTrackerObservation") {
      details = (
        <TaskTrackingObservationContent
          event={event as ObservationEvent<TaskTrackerObservation>}
        />
      );
    } else {
      details =
        getCanvasUIClientObservationContent(event, correspondingAction) ??
        resolveVisualizerBody(event, correspondingAction) ??
        getObservationContent(event);
    }
  } else if (isACPToolCallEvent(event)) {
    // ACP sub-agent tool calls reuse the same card shape as observations:
    // title is "Running/Editing/Reading …" via a translation key that
    // mirrors ACTION_MESSAGE$RUN etc.; details are markdown built from
    // raw_input + raw_output the same way getTerminalObservationContent
    // builds "Command: / Output:" blocks.
    title = createTitleFromKey(getACPToolCallTitleKey(event), {
      // Strip a redundant verb prefix the ACP server may have inlined
      // (Claude Code emits ``"Read /path"`` for a read tool; combined
      // with the ``"Reading <cmd>{{title}}</cmd>"`` template that lands
      // as ``"Reading Read /path"``). See ``stripRedundantTitlePrefix``.
      title: stripRedundantTitlePrefix(event),
    });
    details = getACPToolCallContent(event);
  } else if (
    // Lenient fallback for action-like events that fail the strict isActionEvent() guard
    // (e.g., missing tool_name or tool_call_id). Extract a title from the action kind
    // so the UI shows something meaningful instead of "Unknown event".
    event.source === "agent" &&
    "action" in event &&
    event.action !== null &&
    typeof event.action === "object" &&
    "kind" in event.action &&
    typeof event.action.kind === "string"
  ) {
    title = String(event.action.kind).replace("Action", "").toUpperCase();
  }

  return {
    title: title || i18n.t(I18nKey.EVENT$UNKNOWN_EVENT),
    details,
  };
};
