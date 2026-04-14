import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMProvider } from "#/api/config-service/config-service.types";

async function fetchAllProviders(): Promise<LLMProvider[]> {
  // Providers are a small set; fetch all in one call with a high limit.
  const page = await ConfigService.searchProviders({ limit: 100 });
  return page.items;
}

export const useSearchProviders = () =>
  useQuery({
    queryKey: ["config", "providers"],
    queryFn: fetchAllProviders,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
