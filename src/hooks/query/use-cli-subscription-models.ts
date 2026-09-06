import { useQuery } from "@tanstack/react-query";
import AcpService from "#/api/acp-service/acp-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SUBSCRIPTION_MODELS_QUERY_KEYS } from "#/hooks/query/query-keys";
import type { SubscriptionSource } from "#/utils/subscription-model-catalog";

export function useCliSubscriptionModels(
  source: Extract<SubscriptionSource, "cursor-cli" | "opencode">,
  { enabled = true }: { enabled?: boolean } = {},
) {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";

  return useQuery({
    queryKey: [...SUBSCRIPTION_MODELS_QUERY_KEYS.bySource(source), backend.id],
    queryFn: () => AcpService.listModels(source),
    enabled: enabled && isLocal,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });
}
