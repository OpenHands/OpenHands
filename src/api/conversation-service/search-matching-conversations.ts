import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import AgentServerConversationService from "./agent-server-conversation-service.api";
import {
  CONVERSATION_SEARCH_RECENT_LIMIT,
  CONVERSATION_SEARCH_RESULT_LIMIT,
} from "./conversation-search.constants";

export type SearchMatchingConversationsOptions = {
  signal?: AbortSignal;
};

/**
 * Search conversations by title via the backend `title__contains` filter.
 * Cloud and local agent-server both receive the same wire param
 * (`title__contains`); the JS option is named `titleContains`.
 */
export async function searchMatchingConversations(
  rawQuery: string,
  options: SearchMatchingConversationsOptions = {},
): Promise<AppConversation[]> {
  const query = rawQuery.trim();
  options.signal?.throwIfAborted();

  if (!query) {
    const page = await AgentServerConversationService.searchConversations({
      limit: CONVERSATION_SEARCH_RECENT_LIMIT,
    });
    options.signal?.throwIfAborted();
    return page.items;
  }

  const page = await AgentServerConversationService.searchConversations({
    limit: CONVERSATION_SEARCH_RESULT_LIMIT,
    titleContains: query,
  });
  options.signal?.throwIfAborted();

  return page.items;
}
