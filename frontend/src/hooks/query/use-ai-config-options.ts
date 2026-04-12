import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";

export interface AIConfigOptions {
  /** The recommended default model */
  defaultModel: string;
  /** List of security analyzers available */
  securityAnalyzers: string[];
}

const fetchAiConfigOptions = async (): Promise<AIConfigOptions> => {
  const [modelsResponse, securityAnalyzers] = await Promise.all([
    OptionService.getModels(),
    OptionService.getSecurityAnalyzers(),
  ]);

  return {
    defaultModel: modelsResponse.default_model,
    securityAnalyzers,
  };
};

export const useAIConfigOptions = () =>
  useQuery({
    queryKey: ["ai-config-options"],
    queryFn: fetchAiConfigOptions,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
