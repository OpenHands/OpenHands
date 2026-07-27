import { useQuery } from "@tanstack/react-query";
import {
  VERIFIED_MODELS_GC_TIME,
  VERIFIED_MODELS_QUERY_KEY,
  VERIFIED_MODELS_STALE_TIME,
  fetchVerifiedModelsByProvider,
} from "./use-verified-models";

export const useOpenhandsVerifiedModels = () =>
  useQuery({
    queryKey: VERIFIED_MODELS_QUERY_KEY,
    queryFn: fetchVerifiedModelsByProvider,
    select: (data) => data?.openhands ?? [],
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
