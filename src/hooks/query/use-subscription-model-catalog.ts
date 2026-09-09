import { useMemo } from "react";
import { getAcpProvider } from "#/constants/acp-providers";
import { useAcpAuthStatus } from "#/hooks/query/use-acp-auth-status";
import { useCliSubscriptionModels } from "#/hooks/query/use-cli-subscription-models";
import { useOpenAISubscriptionModels } from "#/hooks/query/use-llm-subscription-models";
import { useOpenAISubscriptionStatus } from "#/hooks/query/use-llm-subscription-status";
import {
  chatgptOffers,
  claudeRegistryOffers,
  countOffersBySource,
  mergeSubscriptionModelOffers,
  type MergedSubscriptionModel,
  type SubscriptionModelOffer,
  type SubscriptionSource,
} from "#/utils/subscription-model-catalog";

export interface SubscriptionModelCatalog {
  offers: SubscriptionModelOffer[];
  rows: MergedSubscriptionModel[];
  countsBySource: Record<SubscriptionSource, number>;
  isLoading: boolean;
}

export function useSubscriptionModelCatalog(): SubscriptionModelCatalog {
  const chatgpt = useOpenAISubscriptionStatus();
  const chatgptConnected = Boolean(chatgpt.data?.connected);
  const chatgptModels = useOpenAISubscriptionModels({
    enabled: chatgptConnected,
  });
  const claude = useAcpAuthStatus("claude-code");
  const cursor = useAcpAuthStatus("cursor-cli");
  const opencode = useAcpAuthStatus("opencode");
  const cursorModels = useCliSubscriptionModels("cursor-cli", {
    enabled: cursor.status === "authenticated",
  });
  const opencodeModels = useCliSubscriptionModels("opencode", {
    enabled: opencode.status === "authenticated",
  });

  const offers = useMemo(() => {
    const next: SubscriptionModelOffer[] = [];
    if (chatgptConnected) {
      next.push(...chatgptOffers(chatgptModels.data ?? []));
    }
    if (claude.status === "authenticated") {
      next.push(
        ...claudeRegistryOffers(
          getAcpProvider("claude-code")?.available_models ?? [],
        ),
      );
    }
    if (cursor.status === "authenticated") {
      next.push(...(cursorModels.data ?? []));
    }
    if (opencode.status === "authenticated") {
      next.push(...(opencodeModels.data ?? []));
    }
    return next;
  }, [
    chatgptConnected,
    chatgptModels.data,
    claude.status,
    cursor.status,
    cursorModels.data,
    opencode.status,
    opencodeModels.data,
  ]);

  const rows = useMemo(() => mergeSubscriptionModelOffers(offers), [offers]);
  const countsBySource = useMemo(() => countOffersBySource(offers), [offers]);

  const isLoading =
    (chatgpt.isFetching && chatgpt.data === undefined) ||
    (chatgptConnected &&
      chatgptModels.isFetching &&
      chatgptModels.data === undefined) ||
    claude.isChecking ||
    cursor.isChecking ||
    opencode.isChecking ||
    Boolean(cursorModels.isFetching && cursorModels.data === undefined) ||
    Boolean(opencodeModels.isFetching && opencodeModels.data === undefined);

  return { offers, rows, countsBySource, isLoading };
}
