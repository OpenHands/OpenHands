import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { filterConversationsByQuery } from "#/utils/conversation-search-filter";
import AgentServerConversationService from "./agent-server-conversation-service.api";
import {
  CONVERSATION_SEARCH_MAX_PAGES,
  CONVERSATION_SEARCH_PAGE_SIZE,
  CONVERSATION_SEARCH_RECENT_LIMIT,
} from "./conversation-search.constants";

/**
 * Search conversations against the backend index, not the sidebar's
 * currently loaded page. When a query is present we paginate through the
 * server search endpoint and apply the existing multi-field match helper
 * so repo/workspace metadata remain searchable on local backends.
 */
export async function searchMatchingConversations(
  rawQuery: string,
): Promise<AppConversation[]> {
  const query = rawQuery.trim();
  if (!query) {
    const page = await AgentServerConversationService.searchConversations({
      limit: CONVERSATION_SEARCH_RECENT_LIMIT,
    });
    return page.items;
  }

  const aggregated: AppConversation[] = [];
  let pageId: string | undefined;

  for (
    let pageIndex = 0;
    pageIndex < CONVERSATION_SEARCH_MAX_PAGES;
    pageIndex++
  ) {
    const page = await AgentServerConversationService.searchConversations({
      limit: CONVERSATION_SEARCH_PAGE_SIZE,
      pageId,
    });
    aggregated.push(...page.items);
    if (!page.next_page_id) {
      break;
    }
    pageId = page.next_page_id;
  }

  return filterConversationsByQuery(aggregated, query);
}
