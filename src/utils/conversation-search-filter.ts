import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

export function getConversationSearchText(
  conversation: AppConversation,
): string {
  return [
    conversation.title,
    conversation.selected_repository,
    conversation.selected_branch,
    conversation.selected_workspace,
    conversation.workspace?.working_dir,
    conversation.llm_model,
  ]
    .filter((part): part is string => !!part && part.trim().length > 0)
    .join(" ");
}

export function conversationMatchesQuery(
  conversation: AppConversation,
  rawQuery: string,
): boolean {
  const query = rawQuery.trim().toLowerCase();
  if (!query) {
    return true;
  }
  return getConversationSearchText(conversation).toLowerCase().includes(query);
}

export function filterConversationsByQuery(
  conversations: readonly AppConversation[],
  rawQuery: string,
): AppConversation[] {
  const query = rawQuery.trim();
  if (!query) {
    return [...conversations];
  }
  return conversations.filter((conversation) =>
    conversationMatchesQuery(conversation, query),
  );
}
