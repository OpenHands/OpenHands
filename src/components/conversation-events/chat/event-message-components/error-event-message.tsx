import React from "react";
import { AgentErrorEvent } from "#/types/agent-server/core";
import { isAgentErrorEvent } from "#/types/agent-server/type-guards";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useAgentState } from "#/hooks/use-agent-state";
import { useIsArchivedConversation } from "#/hooks/use-is-archived-conversation";
import { useResumeConversation } from "#/hooks/mutation/use-resume-conversation";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useEventStore } from "#/stores/use-event-store";
import { AgentState } from "#/types/agent-state";
import { ErrorMessage } from "#/components/features/chat/error-message";

interface ErrorEventMessageProps {
  event: AgentErrorEvent;
}

export function ErrorEventMessage({ event }: ErrorEventMessageProps) {
  const { conversationId } = useOptionalConversationId();
  const { curAgentState } = useAgentState();
  const isArchivedConversation = useIsArchivedConversation();
  const resumeConversationMutation = useResumeConversation();
  const { removeErrorMessage } = useErrorMessageStore();

  const isLatestError = useEventStore((state) => {
    const errorEvents = state.uiEvents.filter(isAgentErrorEvent);
    const lastError = errorEvents.at(-1);
    return lastError ? lastError.id === event?.id : true;
  });

  const handleRetry = React.useCallback(() => {
    if (!conversationId) return;
    resumeConversationMutation.mutate(
      { conversationId },
      {
        onSuccess: () => {
          removeErrorMessage();
        },
      },
    );
  }, [conversationId, removeErrorMessage, resumeConversationMutation]);

  if (!isAgentErrorEvent(event)) {
    return null;
  }

  const isRetryable = event.classification
    ? event.classification.retryable === true ||
      event.classification.user_action === "retry"
    : true;

  const canRetry =
    Boolean(conversationId) &&
    !isArchivedConversation &&
    isLatestError &&
    isRetryable &&
    (curAgentState === AgentState.ERROR ||
      curAgentState === AgentState.RATE_LIMITED ||
      curAgentState === AgentState.STOPPED ||
      curAgentState === AgentState.PAUSED);

  return (
    <ErrorMessage
      errorId={event.id}
      defaultMessage={event.error}
      onRetry={canRetry ? handleRetry : undefined}
      isRetrying={resumeConversationMutation.isPending}
    />
  );
}
