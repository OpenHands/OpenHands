import { getActiveBackend } from "#/api/backend-registry/active-store";

/** Local only: cloud launches do not enforce a profile's secret scope yet. */
export function agentProfileSupportsSecretRefs(): boolean {
  return getActiveBackend().backend.kind !== "cloud";
}

/** Local only: cloud launches ignore a profile's tools for now. */
export function agentProfileSupportsTools(): boolean {
  return getActiveBackend().backend.kind !== "cloud";
}
