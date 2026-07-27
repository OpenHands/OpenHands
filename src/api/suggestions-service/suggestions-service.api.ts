import { SuggestedTask } from "#/utils/types";
import { getActiveBackend } from "../backend-registry/active-store";
import { getCloudSuggestedTasks } from "../cloud/suggestions-service.api";

export class SuggestionsService {
  /**
   * Aggregate suggested tasks for the calling user. Cloud exposes
   * `/api/v1/git/suggested-tasks/search` which the proxy forwards to.
   * Local agent-server has no equivalent endpoint, so we return an
   * empty list there — the home page's `<TaskSuggestions />` already
   * renders nothing when the active backend is local.
   */
  static async getSuggestedTasks(
    pageId?: string,
    limit: number = 30,
  ): Promise<SuggestedTask[]> {
    if (getActiveBackend().backend.kind !== "cloud") {
      return [];
    }
    const page = await getCloudSuggestedTasks({ pageId, limit });
    return page.items;
  }
}
