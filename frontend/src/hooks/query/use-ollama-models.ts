import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";

export const useOllamaModels = (baseUrl: string | null) =>
  useQuery({
    queryKey: ["ollama-models", baseUrl],
    queryFn: () => OptionService.getOllamaModels(baseUrl!),
    enabled: !!baseUrl,
    staleTime: 1000 * 30, // 30 seconds — Ollama models change as users pull/remove
    gcTime: 1000 * 60 * 5, // 5 minutes
    retry: 1,
  });
