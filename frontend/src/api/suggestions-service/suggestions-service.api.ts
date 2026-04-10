import { SuggestedTask } from "#/utils/types";
import { openHands } from "../open-hands-axios";

interface SuggestedTaskPage {
  items: SuggestedTask[];
  next_page_id: string | null;
}

export class SuggestionsService {
  /**
   * Get suggested tasks for the user with pagination.
   *
   * @param pageId - Optional cursor for the next page (from previous response's next_page_id)
   * @param limit - Max number of results per page (default: 30, max: 100)
   */
  static async getSuggestedTasks(
    pageId?: string,
    limit: number = 30,
  ): Promise<SuggestedTask[]> {
    const { data } = await openHands.get<SuggestedTaskPage>(
      "/api/v1/git/suggested-tasks/search",
      {
        params: {
          page_id: pageId ?? undefined,
          limit,
        },
      },
    );
    return data.items;
  }
}
