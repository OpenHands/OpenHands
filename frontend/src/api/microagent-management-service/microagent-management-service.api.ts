import { openHands } from "#/api/open-hands-axios";
import { Conversation, ResultSet } from "#/api/open-hands.types";

class MicroagentManagementService {
  /**
   * Get conversations for microagent management
   * @param selectedRepository The selected repository
   * @param pageId Optional page ID for pagination
   * @param limit Maximum number of conversations to return
   * @returns List of conversations
   */
  static async getMicroagentManagementConversations(
    selectedRepository: string,
    pageId?: string,
    limit: number = 100,
  ): Promise<Conversation[]> {
    const params: Record<string, string | number> = {
      limit,
      selected_repository: selectedRepository,
    };

    if (pageId) {
      params.page_id = pageId;
    }

    const { data } = await openHands.get<ResultSet<Conversation>>(
      "/api/microagent-management/conversations",
      { params },
    );
    return data.results;
  }
}

export default MicroagentManagementService;
