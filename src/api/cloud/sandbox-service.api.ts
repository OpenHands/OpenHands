import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";
import type { V1SandboxInfo } from "./sandbox-service.types";

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud sandboxes call requires a cloud backend.");
  }
  return active;
}

/**
 * Batch-fetch cloud sandboxes by id. Mirrors OpenHands'
 * `SandboxService.batchGetSandboxes` by calling
 * `GET /api/v1/sandboxes?id=...` on the cloud backend, returning each
 * `SandboxInfo` (or null if not found).
 *
 * The returned `SandboxInfo.exposed_urls` carry the cloud-computed,
 * publicly-reachable URLs for the sandbox's services (VSCODE,
 * AGENT_SERVER, WORKER_*) — the GUI reads them directly instead of
 * asking the runtime for `/api/vscode/url`, which only knows its
 * internal localhost address.
 */
export async function batchGetCloudSandboxes(
  ids: string[],
): Promise<(V1SandboxInfo | null)[]> {
  if (ids.length === 0) return [];
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  for (const id of ids) params.append("id", id);
  const data = await callCloudProxy<(V1SandboxInfo | null)[]>({
    backend,
    method: "GET",
    path: `/api/v1/sandboxes?${params.toString()}`,
  });
  return data ?? [];
}
