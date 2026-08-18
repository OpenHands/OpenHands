import React from "react";
import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMModel } from "#/api/config-service/config-service.types";
import type { FreeModelSet } from "#/utils/format-model-name";
import { useFreeModelsStore } from "#/stores/free-models-store";
import {
  VERIFIED_MODELS_GC_TIME,
  VERIFIED_MODELS_QUERY_KEY,
  VERIFIED_MODELS_STALE_TIME,
  fetchVerifiedModelsByProvider,
} from "./use-verified-models";

/**
 * Provider whose models carry free / default metadata. Both are
 * OpenHands-managed concepts, so they are scoped to the `openhands` provider.
 */
const OPENHANDS_PROVIDER = "openhands";

/**
 * Fetches the `openhands` provider's models with their DB-driven `free` /
 * `default` flags (the same channel that carries `verified`).
 */
const useOpenHandsModels = () =>
  useQuery({
    queryKey: ["config", "models", OPENHANDS_PROVIDER, "flags"],
    queryFn: async ({ client }): Promise<LLMModel[]> => {
      const verifiedByProvider = await client.fetchQuery({
        queryKey: VERIFIED_MODELS_QUERY_KEY,
        queryFn: fetchVerifiedModelsByProvider,
        staleTime: VERIFIED_MODELS_STALE_TIME,
      });
      const page = await ConfigService.searchModels(
        { provider__eq: OPENHANDS_PROVIDER, limit: 1000 },
        verifiedByProvider,
      );
      return page.items;
    },
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });

/**
 * Fetches the DB-driven free / default flags once and mirrors them into the
 * {@link useFreeModelsStore}. Mount this high in the tree (inside the query
 * provider). Leaf display components then read the flags synchronously via the
 * {@link useFreeModels} / {@link useDefaultModel} zustand selectors, so they
 * stay renderable in isolation without a QueryClientProvider in scope.
 */
export const useHydrateFreeModels = (): void => {
  const { data } = useOpenHandsModels();
  const setFlags = useFreeModelsStore((state) => state.setFlags);

  React.useEffect(() => {
    if (!data) return;
    const freeModels = new Set(
      data
        .filter((model) => model.free)
        .map((model) => `${OPENHANDS_PROVIDER}/${model.name}`),
    );
    const defaultEntry = data.find((model) => model.default);
    setFlags({
      freeModels,
      defaultModel: defaultEntry
        ? `${OPENHANDS_PROVIDER}/${defaultEntry.name}`
        : null,
    });
  }, [data, setFlags]);
};

/**
 * Set of free ``openhands/<model>`` ids, sourced from the backend model list
 * (DB-driven on cloud via the `free` flag). Returns an empty set on backends
 * without free metadata (e.g. the local agent-server), so callers uniformly
 * treat "not in set" as paid and the frontend keeps no hardcoded free list.
 */
export const useFreeModels = (): FreeModelSet =>
  useFreeModelsStore((state) => state.freeModels);

/**
 * DB-driven default OpenHands model id (``openhands/<model>``), used to
 * preselect a model on onboarding and when creating a new OpenHands model.
 * Returns `null` on backends without default metadata.
 */
export const useDefaultModel = (): string | null =>
  useFreeModelsStore((state) => state.defaultModel);
