import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import OptionService from "#/api/option-service/option-service.api";
import type { SearchModelsParams } from "#/api/config-service/config-service.types";

export interface AIConfigOptions {
  /** All available models (full name with provider prefix) */
  models: string[];
  /** Model names (without provider prefix) that are verified */
  verifiedModels: string[];
  /** Provider names that are verified */
  verifiedProviders: string[];
  /** The recommended default model */
  defaultModel: string;
  /** List of security analyzers available */
  securityAnalyzers: string[];
}

interface SearchResult {
  provider: string | null;
  name: string;
  verified: boolean;
}

const fetchAiConfigOptions = async (): Promise<AIConfigOptions> => {
  // Fetch providers from V1 endpoint
  const providers = await ConfigService.getProviders();

  // Fetch verified models from V1 endpoint
  // We need to fetch all verified models - use a high limit to get them all
  const params: SearchModelsParams = {
    verified__eq: true,
    limit: 100,
  };

  let allModels: SearchResult[] = [];
  let hasMore = true;
  let pageId: string | undefined;

  while (hasMore) {
    if (pageId) {
      params.page_id = pageId;
    }
    const result = await ConfigService.searchModels(params);
    allModels = [...allModels, ...result.items];
    hasMore = result.next_page_id !== null;
    pageId = result.next_page_id ?? undefined;
  }

  // Also fetch non-verified models for the "other" category
  // This gives us a more complete list for the model selector
  const nonVerifiedParams: SearchModelsParams = {
    verified__eq: false,
    limit: 100,
  };

  let nonVerifiedModels: SearchResult[] = [];
  hasMore = true;
  pageId = undefined;

  while (hasMore) {
    if (pageId) {
      nonVerifiedParams.page_id = pageId;
    }
    const result = await ConfigService.searchModels(nonVerifiedParams);
    nonVerifiedModels = [...nonVerifiedModels, ...result.items];
    hasMore = result.next_page_id !== null;
    pageId = result.next_page_id ?? undefined;
  }

  // Combine verified and non-verified, prioritizing verified models first
  const allModelsCombined = [...allModels, ...nonVerifiedModels];

  // Build the models array with full provider/name format
  const models = allModelsCombined
    .map((model) => {
      if (model.provider) {
        return `${model.provider}/${model.name}`;
      }
      return model.name;
    })
    .filter((model, index, self) => self.indexOf(model) === index); // Remove duplicates

  // Extract verified model names (without provider prefix)
  const verifiedModels = allModels.filter((m) => m.verified).map((m) => m.name);

  // Get security analyzers from the deprecated endpoint (still needed)
  // We could also migrate this to V1 if there's a V1 security-analyzers endpoint
  const securityAnalyzers = await OptionService.getSecurityAnalyzers();

  // Get default model from the deprecated endpoint (V1 doesn't have this yet)
  // TODO: Add default_model to V1 endpoint in the future
  const modelsResponse = await OptionService.getModels();

  return {
    models,
    verifiedModels,
    verifiedProviders: providers,
    defaultModel: modelsResponse.default_model,
    securityAnalyzers,
  };
};

export const useAIConfigOptions = () =>
  useQuery({
    queryKey: ["ai-config-options"],
    queryFn: fetchAiConfigOptions,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
