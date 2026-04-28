import { useQuery } from "@tanstack/react-query";
import { llmApi } from "#/api/llm-api";

export interface LLMHealthCheckResponse {
  status:
    | "connected"
    | "timeout"
    | "connection_error"
    | "error"
    | "model_not_found"
    | "configured";
  model: string;
  provider: string;
  latency_ms: number;
  error_message: string;
  is_local: boolean;
}

export function useLLMHealthCheck(enabled = true) {
  return useQuery<LLMHealthCheckResponse>({
    queryKey: ["llm", "health-check"],
    queryFn: async () => {
      const response =
        await llmApi.get<LLMHealthCheckResponse>("/health-check");
      return response.data;
    },
    enabled,
    staleTime: 5000, // 5 seconds
    retry: 0, // Don't auto-retry
  });
}
