import { OpenHandsEvent } from "#/types/v1/core";
import { openHands } from "./open-hands-axios";

export interface SharedConversation {
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

export const sharedConversationService = {
  /**
   * Get a single shared conversation by ID
   */
  async getSharedConversation(
    conversationId: string,
  ): Promise<SharedConversation | null> {
    const response = await openHands.get<(SharedConversation | null)[]>(
      "/api/shared-conversations",
      { params: { ids: conversationId } },
    );
    return response.data[0] || null;
  },

  /**
   * Get events for a shared conversation
   */
  async getSharedConversationEvents(
    conversationId: string,
    limit: number = 100,
    pageId?: string,
  ): Promise<EventPage> {
    const response = await openHands.get<EventPage>(
      "/api/shared-events/search",
      {
        params: {
          conversation_id: conversationId,
          limit,
          ...(pageId && { page_id: pageId }),
        },
      },
    );
    return response.data;
  },
};
