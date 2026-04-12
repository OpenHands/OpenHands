import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMModel } from "#/api/config-service/config-service.types";

async function fetchPage(
  provider: string,
  pageId?: string,
): Promise<LLMModel[]> {
  const page = await ConfigService.searchModels({
    provider__eq: provider,
    limit: 100,
    page_id: pageId,
  });

  if (page.next_page_id) {
    const rest = await fetchPage(provider, page.next_page_id);
    return [...page.items, ...rest];
  }
  return page.items;
}

async function fetchModelsForProvider(provider: string): Promise<LLMModel[]> {
  return fetchPage(provider);
}

export const useProviderModels = (provider: string | null) =>
  useQuery({
    queryKey: ["config", "models", provider],
    queryFn: () => fetchModelsForProvider(provider!),
    enabled: !!provider,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
