import { ObservationMessage } from "#/types/message";
import { useCommandStore } from "#/state/command-store";
import ObservationType from "#/types/observation-type";
import { useBrowserStore } from "#/stores/browser-store";
import { useAgentStore } from "#/stores/agent-store";
import { AgentState } from "#/types/agent-state";

export function handleObservationMessage(message: ObservationMessage) {
  switch (message.observation) {
    case ObservationType.RUN: {
      if (message.extras.hidden) break;
      let { content } = message;

      if (content.length > 5000) {
        const halfLength = 2500;
        const head = content.slice(0, halfLength);
        const tail = content.slice(content.length - halfLength);
        content = `${head}\r\n\n... (truncated ${message.content.length - 5000} characters) ...\r\n\n${tail}`;
      }

      useCommandStore.getState().appendOutput(content);
      break;
    }
    case ObservationType.BROWSE:
    case ObservationType.BROWSE_INTERACTIVE:
      if (
        message.extras?.screenshot &&
        typeof message.extras.screenshot === "string"
      ) {
        useBrowserStore.getState().setScreenshotSrc(message.extras.screenshot);
      }
      if (message.extras?.url && typeof message.extras.url === "string") {
        useBrowserStore.getState().setUrl(message.extras.url);
      }
      break;
    case ObservationType.AGENT_STATE_CHANGED:
      if (typeof message.extras.agent_state === "string") {
        useAgentStore
          .getState()
          .setCurrentAgentState(message.extras.agent_state as AgentState);
      }
      break;
    case ObservationType.DELEGATE:
    case ObservationType.READ:
    case ObservationType.EDIT:
    case ObservationType.THINK:
    case ObservationType.NULL:
    case ObservationType.RECALL:
    case ObservationType.ERROR:
    case ObservationType.MCP:
    case ObservationType.TASK_TRACKING:
      break; // We don't display the default message for these observations
    default:
      break;
  }
  if (!message.extras?.hidden) {
    // Convert the message to the appropriate observation type
    const { observation } = message;

    switch (observation) {
      case "browse":
        if (
          message.extras?.screenshot &&
          typeof message.extras.screenshot === "string"
        ) {
          useBrowserStore
            .getState()
            .setScreenshotSrc(message.extras.screenshot);
        }
        if (message.extras?.url && typeof message.extras.url === "string") {
          useBrowserStore.getState().setUrl(message.extras.url);
        }
        break;
      case "browse_interactive":
        if (
          message.extras?.screenshot &&
          typeof message.extras.screenshot === "string"
        ) {
          useBrowserStore
            .getState()
            .setScreenshotSrc(message.extras.screenshot);
        }
        if (message.extras?.url && typeof message.extras.url === "string") {
          useBrowserStore.getState().setUrl(message.extras.url);
        }
        break;
      default:
        // For any unhandled observation types, just ignore them
        break;
    }
  }
}
