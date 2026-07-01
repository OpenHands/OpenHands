import { openHands } from "./open-hands-axios";
import {
  MarketplaceRegistration,
  MarketplacePluginInfo,
  SkillInfo,
} from "#/types/settings";

interface SkillPage {
  items: SkillInfo[];
  next_page_id: string | null;
}

export interface MarketplaceSkillsResponse {
  skills: SkillInfo[];
  plugins: MarketplacePluginInfo[];
  marketplace_skills: Record<string, string[]>;
  errors: string[];
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

  /**
   * Get skills from marketplace repositories
   *
   * This endpoint fetches and returns skill metadata from marketplace repos
   * without requiring an active sandbox session.
   */
  static async getMarketplaceSkills(
    marketplaces: MarketplaceRegistration[],
  ): Promise<MarketplaceSkillsResponse> {
    const { data } = await openHands.post<MarketplaceSkillsResponse>(
      "/api/v1/skills/marketplace-skills",
      marketplaces,
    );
    return data;
  }
}

export default SkillsService;
