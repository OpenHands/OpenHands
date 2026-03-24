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
    const { data } = await openHands.get<SkillPage>("/api/v1/skills/search", {
      params: { limit: 100 },
    });
    return data.items;
  }
}

export default SkillsService;
