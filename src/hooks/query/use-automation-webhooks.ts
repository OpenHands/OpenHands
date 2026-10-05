import { useRef } from "react";
import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import type { CreateAutomationWebhookRequest } from "#/types/automation-webhook";
import { getErrorStatus } from "./use-settings";

const PAGE_SIZE = 50;

export const AUTOMATION_WEBHOOKS_QUERY_KEYS = {
  all: ["automation-webhooks"] as const,
  forBackend: (
    backendId: string,
    orgId: string | null | undefined,
    revision: number,
  ) => ["automation-webhooks", backendId, orgId, revision] as const,
};

// @spec BM-002 — Custom event sources
export function useAutomationWebhooks(enabled: boolean) {
  const { backend, orgId } = useActiveBackend();
  return useInfiniteQuery({
    queryKey: AUTOMATION_WEBHOOKS_QUERY_KEYS.forBackend(
      backend.id,
      orgId,
      backend.connectionRevision ?? 0,
    ),
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      AutomationService.listWebhooks({ limit: PAGE_SIZE, offset: pageParam }),
    getNextPageParam: (lastPage, pages) => {
      const loaded = pages.reduce(
        (count, page) => count + page.webhooks.length,
        0,
      );
      return loaded < lastPage.total && lastPage.webhooks.length > 0
        ? loaded
        : undefined;
    },
    enabled,
    retry: false,
    meta: { disableToast: true },
  });
}

/** No secret enters React Query's variables or result cache. */
export function useCreateAutomationWebhook(
  onGeneratedSecret: (secret: string) => void,
) {
  const { backend, orgId } = useActiveBackend();
  const client = useQueryClient();
  const pending = useRef<CreateAutomationWebhookRequest | null>(null);
  const queryKey = AUTOMATION_WEBHOOKS_QUERY_KEYS.forBackend(
    backend.id,
    orgId,
    backend.connectionRevision ?? 0,
  );
  const mutation = useMutation({
    mutationFn: async () => {
      const body = pending.current;
      if (!body) throw new Error("No webhook registration is pending");
      try {
        const { webhook_secret: generatedSecret, ...record } =
          await AutomationService.createWebhook(body);
        if (generatedSecret) onGeneratedSecret(generatedSecret);
        return record;
      } catch (error) {
        // Axios errors retain the request body (including a supplied secret).
        // Keep only the status in the mutation cache.
        throw Object.assign(new Error("Webhook registration failed"), {
          status: getErrorStatus(error),
        });
      } finally {
        pending.current = null;
      }
    },
    onSuccess: () => client.invalidateQueries({ queryKey }),
    meta: { disableToast: true },
  });

  const create = (body: CreateAutomationWebhookRequest) => {
    if (pending.current || mutation.isPending) return;
    pending.current = body;
    mutation.mutate();
  };
  return { ...mutation, create };
}
