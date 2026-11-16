import React from "react";
import { AgentErrorEvent } from "#/types/agent-server/core";
import { isAgentErrorEvent } from "#/types/agent-server/type-guards";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useAgentState } from "#/hooks/use-agent-state";
import { useResumeConversation } from "#/hooks/mutation/use-resume-conversation";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useEventStore } from "#/stores/use-event-store";
import { AgentState } from "#/types/agent-state";
import { ErrorMessage } from "../../../features/chat/error-message";

interface ErrorEventMessageProps {
  event: AgentErrorEvent;
}

export function ErrorEventMessage({ event }: ErrorEventMessageProps) {
  if (!isAgentErrorEvent(event)) {
    return null;
  }

  const { conversationId } = useOptionalConversationId();
  const { curAgentState } = useAgentState();
  const resumeConversationMutation = useResumeConversation();
  const { removeErrorMessage } = useErrorMessageStore();

  const isLatestError = useEventStore((state) => {
    const errorEvents = state.uiEvents.filter(isAgentErrorEvent);
    const lastError = errorEvents.at(-1);
    return lastError ? lastError.id === event.id : true;
  });

  const canRetry =
    Boolean(conversationId) &&
    isLatestError &&
    (curAgentState === AgentState.ERROR ||
      curAgentState === AgentState.RATE_LIMITED ||
      curAgentState === AgentState.STOPPED ||
      curAgentState === AgentState.PAUSED);

  const handleRetry = React.useCallback(() => {
    if (!conversationId) return;
    removeErrorMessage();
    resumeConversationMutation.mutate({ conversationId });
  }, [conversationId, removeErrorMessage, resumeConversationMutation]);

  return (
    <ErrorMessage
      errorId={event.id}
      defaultMessage={event.error}
      onRetry={canRetry ? handleRetry : undefined}
      isRetrying={resumeConversationMutation.isPending}
    />
  );
}
