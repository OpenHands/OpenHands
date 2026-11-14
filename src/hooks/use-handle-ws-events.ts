import React from "react";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useEventStore } from "#/stores/use-event-store";
import { usePauseConversation } from "#/hooks/mutation/use-pause-conversation";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  isAgentErrorEvent,
  isAgentServerEvent,
  isConversationErrorEvent,
  isHookExecutionEvent,
} from "#/types/agent-server/type-guards";

interface ServerError {
  error: boolean | string;
  message: string;
  [key: string]: unknown;
}

const isServerError = (data: object): data is ServerError => "error" in data;

/**
 * Structured code the agent server sets on the ConversationErrorEvent it
 * emits when a run hits its iteration cap (software-agent-sdk:
 * openhands/sdk/conversation/impl/local_conversation.py). The event's
 * `detail` text is display copy; the code is the stable signal.
 */
const MAX_ITERATIONS_REACHED_CODE = "MaxIterationsReached";

export const useHandleWSEvents = () => {
  const events = useEventStore((state) => state.events);
  const { conversationId } = useOptionalConversationId();
  const pauseConversation = usePauseConversation();

  // The store is hydrated from conversation history, so the first snapshot
  // this hook sees is the past, not a live arrival: a max-iterations error
  // replayed from an earlier session must not pause the conversation the
  // user just reopened. Only events arriving after that snapshot count.
  const hasInitialSnapshotRef = React.useRef(false);

  React.useEffect(() => {
    if (!events.length) {
      return;
    }
    const event = events[events.length - 1];
    const isInitialSnapshot = !hasInitialSnapshotRef.current;
    hasInitialSnapshotRef.current = true;

    // V1 agent errors are surfaced inline in the chat log (and via the error
    // banner), so don't double-notify with a toast. Hook executions render
    // their own inline row too, and a persisted hook event carries
    // `error: null`, which the legacy server-error check below would
    // otherwise mistake for a failure with no message.
    if (
      isAgentServerEvent(event) &&
      (isAgentErrorEvent(event) || isHookExecutionEvent(event))
    ) {
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

    if (
      !isInitialSnapshot &&
      isConversationErrorEvent(event) &&
      event.code === MAX_ITERATIONS_REACHED_CODE &&
      !(event as { isFromPlanningAgent?: boolean }).isFromPlanningAgent &&
      conversationId
    ) {
      // The run already ended in an error state on the server. Pausing
      // through the supported mutation holds the conversation (and, on
      // cloud backends, the sandbox) in a resumable paused state instead
      // of leaving it in the error state. Planning-agent events share
      // this store; a sub-conversation hitting its own cap must not
      // pause the main conversation.
      pauseConversation.mutate({ conversationId });
    }
  }, [events.length]);
};
