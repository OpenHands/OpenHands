import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";

const fetchAiConfigOptions = async () => {
  const modelsResponse = await OptionService.getModels();
  return {
    models: modelsResponse.models,
    verifiedModels: modelsResponse.verified_models,
    verifiedProviders: modelsResponse.verified_providers,
    defaultModel: modelsResponse.default_model,
    securityAnalyzers: await OptionService.getSecurityAnalyzers(),
  };
};

export const useAIConfigOptions = () =>
  useQuery({
    queryKey: ["ai-config-options"],
    queryFn: fetchAiConfigOptions,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
