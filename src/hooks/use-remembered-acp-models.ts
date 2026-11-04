import { useMemo, useSyncExternalStore } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  parseRememberedAcpModels,
  readRememberedAcpModelsRaw,
  subscribeRememberedAcpModels,
} from "#/utils/remembered-acp-models";

/**
 * The models ``providerKey`` last reported on this backend under ``scope``, or
 * under any scope when omitted.
 */
export function useRememberedAcpModels(
  providerKey: string | null | undefined,
  scope?: string,
) {
  const { backend } = useActiveBackend();
  const raw = useSyncExternalStore(
    subscribeRememberedAcpModels,
    () => readRememberedAcpModelsRaw(backend.id, providerKey),
    () => null,
  );
  return useMemo(() => parseRememberedAcpModels(raw, scope), [raw, scope]);
}
