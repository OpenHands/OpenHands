import { useMutation } from "@tanstack/react-query";
import SkillsService, { MarketplaceSkillsResponse } from "#/api/skills-service";
import { MarketplaceRegistration } from "#/types/settings";

export const useMarketplaceSkills = () =>
  useMutation({
    mutationFn: async (
      marketplaces: MarketplaceRegistration[],
    ): Promise<MarketplaceSkillsResponse> =>
      SkillsService.getMarketplaceSkills(marketplaces),
  });
