import { useMemo } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { readRememberedAcpModels } from "#/utils/remembered-acp-models";

/** The models ``providerKey`` last reported in a conversation on this backend. */
export function useRememberedAcpModels(providerKey: string | null | undefined) {
  const { backend } = useActiveBackend();
  return useMemo(
    () => readRememberedAcpModels(backend.id, providerKey),
    [backend.id, providerKey],
  );
}
