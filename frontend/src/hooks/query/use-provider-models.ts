import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMModel } from "#/api/config-service/config-service.types";

const MAX_PAGINATION_DEPTH = 10;

async function fetchPage(
  provider: string,
  pageId?: string,
  depth = 0,
): Promise<LLMModel[]> {
  if (depth >= MAX_PAGINATION_DEPTH) {
    throw new Error(`Too many pagination requests for provider ${provider}`);
  }

  const page = await ConfigService.searchModels({
    provider__eq: provider,
    limit: 100,
    page_id: pageId,
  });

  if (page.next_page_id) {
    const rest = await fetchPage(provider, page.next_page_id, depth + 1);
    return [...page.items, ...rest];
  }
  return page.items;
}

export const useProviderModels = (provider: string | null) =>
  useQuery({
    queryKey: ["config", "models", provider],
    queryFn: () => fetchPage(provider!),
    enabled: !!provider,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
