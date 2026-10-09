import { useMemo, useSyncExternalStore } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  parseRememberedAcpModels,
  readRememberedAcpModelsRaw,
  subscribeRememberedAcpModels,
} from "#/utils/remembered-acp-models";

/** The models ``providerKey`` last reported in a conversation on this backend. */
export function useRememberedAcpModels(providerKey: string | null | undefined) {
  const { backend } = useActiveBackend();
  const raw = useSyncExternalStore(
    subscribeRememberedAcpModels,
    () => readRememberedAcpModelsRaw(backend.id, providerKey),
    () => null,
  );
  return useMemo(() => parseRememberedAcpModels(raw), [raw]);
}
