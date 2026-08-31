import { useQuery } from "@tanstack/react-query";
import LLMBalanceService from "#/api/llm-balance-service";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * Provider credit balance for the active local agent server.
 *
 * Fetched once per conversation load (the query key includes the
 * conversation id); use `refetch()` for the manual refresh affordance.
 * `data === null` means the server/provider doesn't support balance
 * reporting (404) — hide the UI. Errors also hide the UI (`retry: false`,
 * consumers check `isError`).
 */
export const useLLMBalance = (conversationId: string | null | undefined) => {
  const { backend } = useActiveBackend();

  return useQuery({
    queryKey: ["llm-balance", backend.id, conversationId],
    queryFn: () => LLMBalanceService.getBalance(),
    // Cloud conversations talk to the hosted app API, which has no
    // /api/llm/balance endpoint — don't probe it.
    enabled: backend.kind !== "cloud" && !!conversationId,
    staleTime: Infinity,
    gcTime: 1000 * 60 * 5,
    retry: false,
    refetchOnWindowFocus: false,
  });
};
