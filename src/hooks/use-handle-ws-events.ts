import React from "react";
import { generateAgentStateChangeEvent } from "#/services/agent-state-service";
import { AgentState } from "#/types/agent-state";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useEventStore } from "#/stores/use-event-store";
import { useSendMessage } from "#/hooks/use-send-message";
import {
  isAgentErrorEvent,
  isAgentServerEvent,
} from "#/types/agent-server/type-guards";

interface ServerError {
  error: boolean | string;
  message: string;
  [key: string]: unknown;
}

const isServerError = (data: object): data is ServerError => "error" in data;

const isTypedErrorEvent = (
  event: object,
): event is { type: "error"; message?: unknown } =>
  "type" in event && event.type === "error";

export const useHandleWSEvents = () => {
  const { send } = useSendMessage();
  const events = useEventStore((state) => state.events);

  React.useEffect(() => {
    if (!events.length) {
      return;
    }
    const event = events[events.length - 1];

    // V1 agent errors are surfaced inline in the chat log (and via the error
    // banner), so don't double-notify with a toast.
    if (isAgentServerEvent(event) && isAgentErrorEvent(event)) {
      return;
    }

    if (isServerError(event)) {
      if (event.error_code === 401) {
        displayErrorToast("Session expired.");
        return;
      }

      if (typeof event.error === "string") {
        displayErrorToast(event.error);
      } else {
        displayErrorToast(event.message);
      }
      return;
    }

    if (isTypedErrorEvent(event)) {
      const message: string = `${event.message ?? ""}`;
      if (message.startsWith("Agent reached maximum")) {
        // We set the agent state to paused here - if the user clicks resume, it auto updates the max iterations
        send(generateAgentStateChangeEvent(AgentState.PAUSED));
      }
    }
  }, [events.length]);
};
