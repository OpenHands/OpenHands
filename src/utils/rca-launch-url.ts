import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { RcaContext } from "#/utils/rca-context";

/**
 * Build the in-app path to the `/launch` screen carrying structured RCA
 * context alongside the required plugins, so external systems (HolmesGPT or
 * any other RCA producer) can deep-link a user into a conversation prefilled
 * with the analysis. Inverse of the `rca` decoding in
 * `src/routes/launch.tsx` — keep the base64-encoded JSON `rca` format in
 * sync with `decodeRcaParam` in `#/utils/rca-context`.
 */
export function buildRcaLaunchPath(
  rca: RcaContext,
  plugins: PluginSpec[],
  message?: string,
): string {
  const params = new URLSearchParams({
    plugins: btoa(JSON.stringify(plugins)),
    rca: btoa(JSON.stringify(rca)),
  });
  if (message) {
    params.set("message", message);
  }
  return `/launch?${params.toString()}`;
}
