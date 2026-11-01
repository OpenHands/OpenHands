import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useGoalStore } from "#/stores/goal-store";

/**
 * Hand a client tool's outcome back to the agent.
 *
 * Client tools are acknowledged by the agent-server before the browser does any
 * work, so a prefixed `user` message is the only result channel. Sending it
 * runs the agent, which relays the outcome in its next turn; the chat hides the
 * raw message (see `should-render-event.ts`).
 *
 * The agent-server cancels an active `/goal` loop on any inbound message, so
 * nothing is posted while one runs; callers surface the outcome to the user
 * themselves.
 */
export async function postClientToolResult(
  conversationId: string,
  prefix: string,
  result: unknown,
): Promise<void> {
  const goalStatus =
    useGoalStore.getState().statusByConversation[conversationId];
  if (goalStatus?.active) return;

  await AgentServerConversationService.sendMessage(conversationId, {
    role: "user",
    content: [{ type: "text", text: `${prefix}${JSON.stringify(result)}` }],
  });
}
