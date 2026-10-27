import { useQuery } from "@tanstack/react-query";
import {
  getMaritimeBridge,
  requireMaritimeBridge,
} from "#/api/maritime/maritime-bridge";
import { MARITIME_QUERY_KEYS } from "./query-keys";

/** Agents sleep, wake and finish deploying on their own; keep the list honest. */
const REFETCH_INTERVAL_MS = 10_000;

/** Whether this Canvas host holds a Maritime API key. Never the key itself. */
export function useMaritimeAuthState() {
  return useQuery({
    queryKey: MARITIME_QUERY_KEYS.authState,
    queryFn: () => requireMaritimeBridge().getAuthState(),
    enabled: getMaritimeBridge() !== null,
    // Sign-in and sign-out invalidate MARITIME_QUERY_KEYS.all; a sign-in from
    // another tab (the web build shares the server's state) does not.
    staleTime: 60_000,
    retry: false,
    meta: { disableToast: true },
  });
}

/** The plan and the memory range a new agent may use. */
export function useMaritimeAccount(isSignedIn: boolean) {
  return useQuery({
    queryKey: MARITIME_QUERY_KEYS.account,
    queryFn: () => requireMaritimeBridge().getAccount(),
    enabled: isSignedIn && getMaritimeBridge() !== null,
    staleTime: 60_000,
    retry: false,
    meta: { disableToast: true },
  });
}

/** The OpenHands agents Canvas created on the signed-in Maritime account. */
export function useMaritimeAgents(
  isSignedIn: boolean,
  {
    refetchIntervalMs = REFETCH_INTERVAL_MS,
  }: { refetchIntervalMs?: number } = {},
) {
  return useQuery({
    queryKey: MARITIME_QUERY_KEYS.agents,
    queryFn: () => requireMaritimeBridge().listAgents(),
    enabled: isSignedIn && getMaritimeBridge() !== null,
    refetchInterval: refetchIntervalMs,
    staleTime: refetchIntervalMs,
    retry: false,
    meta: { disableToast: true },
  });
}
