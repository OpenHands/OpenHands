import { getDefaultConversationTitle } from "#/api/agent-server-adapter";

export function isPlaceholderConversationTitle(
  title: string | null | undefined,
  conversationId: string,
): boolean {
  return (
    !title?.trim() ||
    title.trim() === getDefaultConversationTitle(conversationId)
  );
}
