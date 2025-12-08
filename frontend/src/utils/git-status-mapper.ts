import type {
  GitChangeStatus,
  V1GitChangeStatus,
} from "#/api/open-hands.types";

/**
 * Maps V1 git change status to legacy V0 status format
 *
 * V1 -> V0 mapping:
 * - ADDED -> A (Added)
 * - DELETED -> D (Deleted)
 * - UPDATED -> M (Modified)
 * - MOVED -> R (Renamed)
 *
 * @param v1Status The V1 git change status
 * @returns The equivalent V0 git change status
 */
export function mapV1ToV0Status(v1Status: V1GitChangeStatus): GitChangeStatus {
  const statusMap: Record<V1GitChangeStatus, GitChangeStatus> = {
    ADDED: "A",
    DELETED: "D",
    UPDATED: "M",
    MOVED: "R",
  };

  return statusMap[v1Status];
}
