import { useQuery } from "@tanstack/react-query";
import SkillsService from "#/api/skills-service";
import { SkillInfo } from "#/types/settings";

export const useSkills = () =>
  useQuery<SkillInfo[]>({
    queryKey: ["skills"],
    queryFn: SkillsService.getSkills,
    staleTime: 1000 * 60 * 10, // 10 minutes – skill list rarely changes
    refetchOnWindowFocus: false,
  });
