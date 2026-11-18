import {
  ASK_USER_LEDGER_STORAGE_KEY_PREFIX,
  ASK_USER_RESULT_PREFIX,
  ASK_USER_TOOL_NAME,
} from "#/constants/ask-user";
import {
  type AskUserDecisionRequest,
  useAskUserStore,
} from "#/stores/ask-user-store";
import type { AskUserAction } from "#/types/agent-server/core";
import { claimClientToolCall } from "#/utils/claim-client-tool-call";
import { postClientToolResult } from "./client-tool-result";

/**
 * Queue an interactive decision card; intercepted from WebSocket ActionEvent.
 * Deduplicated via claimClientToolCall so replayed events do not queue again.
 */
export async function handleAskUserAction(
  action: AskUserAction,
  conversationId: string,
  toolCallId: string,
): Promise<void> {
  // A replayed call would ask again and report the outcome twice.
  if (
    !claimClientToolCall(
      `${ASK_USER_LEDGER_STORAGE_KEY_PREFIX}${conversationId}`,
      toolCallId,
    )
  ) {
    return;
  }

  const request: AskUserDecisionRequest = {
    conversationId,
    toolCallId,
    question: action.question || "",
    options: Array.isArray(action.options) ? action.options : [],
    recommended: action.recommended ?? null,
    selectedAnswer: null,
  };
  useAskUserStore.getState().enqueue(request);
}

/**
 * Hand the outcome of a user decision back to the agent.
 * Updates the store to record the answered state and posts the client tool result.
 */
export async function completeAskUserDecision(
  request: AskUserDecisionRequest,
  answer: string,
): Promise<void> {
  const store = useAskUserStore.getState();
  store.answer(request.toolCallId, answer);

  await postClientToolResult(request.conversationId, ASK_USER_RESULT_PREFIX, {
    toolCallId: request.toolCallId,
    answer,
  }).catch((error) => {
    console.warn(
      `[${ASK_USER_TOOL_NAME}] Failed to report the decision result:`,
      error,
    );
  });
}
