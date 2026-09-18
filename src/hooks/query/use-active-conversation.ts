import { useEffect, useRef } from "react";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useUserConversation } from "./use-user-conversation";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { isExecutionActive } from "#/utils/status";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

const FAST_POLL_MS = 3000;
const SLOW_POLL_MS = 30000;

/** Untitled IDLE/FINISHED conversations get this many 3s polls, then 30s. */
export const UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS = 5;

const TERMINAL_EXECUTION_STATUSES: ReadonlySet<ExecutionStatus> = new Set([
  ExecutionStatus.IDLE,
  ExecutionStatus.FINISHED,
]);

function isTerminalExecutionStatus(
  status: ExecutionStatus | null | undefined,
): boolean {
  return !!status && TERMINAL_EXECUTION_STATUSES.has(status);
}

function nextActiveConversationPollMs(
  data: AppConversation | null | undefined,
  untitledTerminalAttempts: number,
): { interval: number; untitledTerminalAttempts: number } {
  if (!data) {
    return { interval: SLOW_POLL_MS, untitledTerminalAttempts: 0 };
  }

  // Wake-up / resume: always fast-poll, and do not consume the title budget.
  if (!data.conversation_url || data.sandbox_status === "PAUSED") {
    return { interval: FAST_POLL_MS, untitledTerminalAttempts: 0 };
  }

  if (!data.title && isExecutionActive(data.execution_status)) {
    if (isTerminalExecutionStatus(data.execution_status)) {
      const nextAttempts = untitledTerminalAttempts + 1;
      return {
        interval:
          nextAttempts > UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS
            ? SLOW_POLL_MS
            : FAST_POLL_MS,
        untitledTerminalAttempts: nextAttempts,
      };
    }
    // Still running: keep fast-polling until a title lands or execution ends.
    return { interval: FAST_POLL_MS, untitledTerminalAttempts: 0 };
  }

  return { interval: SLOW_POLL_MS, untitledTerminalAttempts: 0 };
}

export const useActiveConversation = () => {
  // Optional: the chat input renders on the home page too (no conversation
  // route yet). The user-conversation query is gated on a real id below.
  const { conversationId } = useOptionalConversationId();

  // Task polling is handled by useTaskPolling hook
  const isTaskId = !!conversationId && conversationId.startsWith("task-");
  const actualConversationId =
    !conversationId || isTaskId ? null : conversationId;

  const untitledTerminalAttemptsRef = useRef(0);
  const countedConversationIdRef = useRef(actualConversationId);
  if (countedConversationIdRef.current !== actualConversationId) {
    countedConversationIdRef.current = actualConversationId;
    untitledTerminalAttemptsRef.current = 0;
  }

  const userConversation = useUserConversation(
    actualConversationId,
    // Fast-poll (3 s) while: the sandbox URL is absent; the sandbox is PAUSED
    // (it keeps the stale conversation_url, so a missing-URL check alone misses
    // the wake-up); or the agent is executing but has no title yet (the title
    // lands asynchronously after conversation_url is already set). Untitled
    // IDLE/FINISHED conversations give up after
    // UNTITLED_TERMINAL_FAST_POLL_ATTEMPTS so a title that never lands does
    // not poll at 3 s indefinitely.
    (query) => {
      const result = nextActiveConversationPollMs(
        query.state.data,
        untitledTerminalAttemptsRef.current,
      );
      untitledTerminalAttemptsRef.current = result.untitledTerminalAttempts;
      return result.interval;
    },
  );

  useEffect(() => {
    const conversation = userConversation.data;
    ConversationService.setCurrentConversation(conversation || null);
  }, [
    conversationId,
    userConversation.isFetched,
    userConversation?.data?.execution_status,
  ]);
  return userConversation;
};
