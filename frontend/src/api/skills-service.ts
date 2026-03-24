import { openHands } from "./open-hands-axios";
import { SkillInfo } from "#/types/settings";

interface SkillPage {
  items: SkillInfo[];
  next_page_id: string | null;
}

class SkillsService {
  /**
   * Search available skills (global + user skills) with pagination
   */
  static async getSkills(): Promise<SkillInfo[]> {
    const allSkills: SkillInfo[] = [];
    let pageId: string | null = null;

    do {
      const params: Record<string, string | number> = { limit: 100 };
      if (pageId) params.page_id = pageId;

      const { data } = await openHands.get<SkillPage>(
        "/api/v1/skills/search",
        { params },
      );
      allSkills.push(...data.items);
      pageId = data.next_page_id;
    } while (pageId);

    return allSkills;
  }
}

export default SkillsService;
