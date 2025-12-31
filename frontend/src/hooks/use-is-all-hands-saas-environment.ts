import { useMemo } from "react";

/**
 * Hook to check if the current domain is an All Hands SaaS environment
 * @returns True if the current domain contains "all-hands.dev" or "openhands.dev" postfix
 */
export const useIsAllHandsSaaSEnvironment = (): boolean =>
  useMemo(() => {
    const { hostname } = window.location;
    return (
      hostname.endsWith("all-hands.dev") || hostname.endsWith("openhands.dev")
    );
  }, []);
