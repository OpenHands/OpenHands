import { Trans } from "react-i18next";
import React from "react";
import { OpenHandsEvent, ObservationEvent } from "#/types/v1/core";
import { isActionEvent, isObservationEvent } from "#/types/v1/type-guards";
import { MonoComponent } from "../../../features/chat/mono-component";
import { PathComponent } from "../../../features/chat/path-component";
import { getActionContent } from "./get-action-content";
import { getObservationContent } from "./get-observation-content";
import { TaskTrackingObservationContent } from "../task-tracking/task-tracking-observation-content";
import { TaskTrackerObservation } from "#/types/v1/core/base/observation";
import { SkillReadyEvent, isSkillReadyEvent } from "./create-skill-ready-event";
import i18n from "#/i18n";

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
      i18nKey={key}
      values={values}
      components={{
        path: <PathComponent />,
        cmd: <MonoComponent />,
      }}
    />
  );
};

// Action Event Processing
const getActionEventTitle = (event: OpenHandsEvent): React.ReactNode => {
  // Early return if not an action event
  if (!isActionEvent(event)) {
    return "";
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
    case "ThinkAction":
      actionKey = "ACTION_MESSAGE$THINK";
      break;
    case "FinishAction":
      actionKey = "ACTION_MESSAGE$FINISH";
      break;
    case "TaskTrackerAction":
      actionKey = "ACTION_MESSAGE$TASK_TRACKING";
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
const getObservationEventTitle = (event: OpenHandsEvent): React.ReactNode => {
  // Early return if not an observation event
  if (!isObservationEvent(event)) {
    return "";
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
    default:
      // For unknown observations, use the type name
      return observationType.replace("Observation", "").toUpperCase();
  }

  if (observationKey) {
    return createTitleFromKey(observationKey, observationValues);
  }

  return observationType;
};

export const getEventContent = (event: OpenHandsEvent | SkillReadyEvent) => {
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
    details = getActionContent(event);
  } else if (isObservationEvent(event)) {
    title = getObservationEventTitle(event);

    // For TaskTrackerObservation, use React component instead of markdown
    if (event.observation.kind === "TaskTrackerObservation") {
      details = (
        <TaskTrackingObservationContent
          event={event as ObservationEvent<TaskTrackerObservation>}
        />
      );
    } else {
      details = getObservationContent(event);
    }
  }

  return {
    title: title || i18n.t("EVENT$UNKNOWN_EVENT"),
    details: details || i18n.t("EVENT$UNKNOWN_EVENT"),
  };
};
