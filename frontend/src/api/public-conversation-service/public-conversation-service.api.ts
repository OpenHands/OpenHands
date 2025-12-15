import { openHands } from "#/api/open-hands-axios";
import { OpenHandsEvent } from "#/types/v1/core/openhands-event";

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
  metrics: Record<string, unknown> | null;
  parent_conversation_id: string | null;
  sub_conversation_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface EventPage {
  events: OpenHandsEvent[];
  next_page_id: string | null;
}

class PublicConversationService {
  /**
   * Get a public conversation by ID
   * @param conversationId The conversation ID
   * @returns The public conversation or null if not found/not public
   */
  static async getPublicConversation(
    conversationId: string,
  ): Promise<PublicConversation | null> {
    try {
      const { data } = await openHands.get<(PublicConversation | null)[]>(
        `/api/v1/public-conversations?ids=${conversationId}`,
      );
      return data[0] || null;
    } catch {
      return null;
    }
  }

  /**
   * Get events for a public conversation
   * @param conversationId The conversation ID
   * @param limit Maximum number of events to fetch
   * @returns The events page
   */
  static async getPublicConversationEvents(
    conversationId: string,
    limit: number = 100,
  ): Promise<EventPage> {
    const { data } = await openHands.get<EventPage>(
      `/api/v1/public-events/search?conversation_id=${conversationId}&limit=${limit}&sort_order=TIMESTAMP`,
    );
    return data;
  }
}

export default PublicConversationService;
