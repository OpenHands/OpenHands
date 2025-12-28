// sandbox-service.api.ts
// This file contains API methods for /api/v1/sandboxes endpoints.

import { openHands } from "../open-hands-axios";
import type { V1SandboxInfo } from "./sandbox-service.types";

export class SandboxService {
  /**
   * Pause a V1 sandbox
   * Calls the /api/v1/sandboxes/{id}/pause endpoint
   */
  static async pauseSandbox(sandboxId: string): Promise<{ success: boolean }> {
    const { data } = await openHands.post<{ success: boolean }>(
      `/api/v1/sandboxes/${sandboxId}/pause`,
      {},
    );
    return data;
  }

  /**
   * Resume a V1 sandbox
   * Calls the /api/v1/sandboxes/{id}/resume endpoint
   */
  static async resumeSandbox(sandboxId: string): Promise<{ success: boolean }> {
    const { data } = await openHands.post<{ success: boolean }>(
      `/api/v1/sandboxes/${sandboxId}/resume`,
      {},
    );
    return data;
  }

  /**
   * Batch get V1 sandboxes by their IDs
   * Returns null for any missing sandboxes
   */
  static async batchGetSandboxes(
    ids: string[],
  ): Promise<(V1SandboxInfo | null)[]> {
    if (ids.length === 0) {
      return [];
    }
    if (ids.length > 100) {
      throw new Error("Cannot request more than 100 sandboxes at once");
    }
    const params = new URLSearchParams();
    ids.forEach((id) => params.append("id", id));
    const { data } = await openHands.get<(V1SandboxInfo | null)[]>(
      `/api/v1/sandboxes?${params.toString()}`,
    );
    return data;
  }
}
