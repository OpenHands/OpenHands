import type { ResolvedActiveBackend } from "#/api/backend-registry/types";

// @spec BM-002 — Home selections belong to one backend connection and organization
export function getHomeLaunchScope({
  backend,
  orgId,
}: ResolvedActiveBackend): string {
  return JSON.stringify([
    backend.id,
    backend.kind,
    backend.connectionRevision ?? 0,
    orgId ?? null,
  ]);
}
