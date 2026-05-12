import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";

export const useAIConfigOptions = () =>
  useQuery({
    queryKey: ["ai-config-options"],
    queryFn: OptionService.getSecurityAnalyzers,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
