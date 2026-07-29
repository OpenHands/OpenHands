import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { filterConversationsByQuery } from "#/utils/conversation-search-filter";
import AgentServerConversationService from "./agent-server-conversation-service.api";
import {
  CONVERSATION_SEARCH_RECENT_LIMIT,
  CONVERSATION_SEARCH_RESULT_LIMIT,
} from "./conversation-search.constants";

export type SearchMatchingConversationsOptions = {
  signal?: AbortSignal;
};

/**
 * Search conversations via the backend title filter, not by downloading the
 * full conversation index. Cloud uses `title__contains`; local agent-server
 * uses the same `titleContains` param. Client-side multi-field filtering still
 * runs on the returned page so repo/workspace/model tokens refine matches.
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

  return filterConversationsByQuery(page.items, query);
}
