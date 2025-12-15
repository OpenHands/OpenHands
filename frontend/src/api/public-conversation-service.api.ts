import { OpenHandsEvent } from "#/types/v1/core";
import { openHands } from "./open-hands-axios";

export interface PublicConversation {
  id: string;
  created_by_user_id: string | null;
  sandbox_id: string;
  selected_repository: string | null;
  selected_branch: string | null;
  git_provider: string | null;
  title: string | null;
  pr_number: number[];
  llm_model: string | null;
  metrics: unknown | null;
  parent_conversation_id: string | null;
  sub_conversation_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface EventPage {
  items: OpenHandsEvent[];
  next_page_id: string | null;
}

export const publicConversationService = {
  /**
   * Get a single public conversation by ID
   */
  async getPublicConversation(
    conversationId: string,
  ): Promise<PublicConversation | null> {
    const response = await openHands.get(
      `/api/v1/public-conversations?ids=${conversationId}`,
    );
    const conversations = response.data as (PublicConversation | null)[];
    return conversations[0] || null;
  },

  /**
   * Get events for a public conversation
   */
  async getPublicConversationEvents(
    conversationId: string,
    limit: number = 100,
    pageId?: string,
  ): Promise<EventPage> {
    const params = new URLSearchParams({
      conversation_id: conversationId,
      limit: limit.toString(),
    });

    if (pageId) {
      params.append("page_id", pageId);
    }

    const response = await openHands.get(
      `/api/v1/public-events/search?${params.toString()}`,
    );
    return response.data as EventPage;
  },
};
