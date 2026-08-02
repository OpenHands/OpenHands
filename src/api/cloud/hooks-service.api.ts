import type { GetHooksResponse } from "#/api/conversation-service/agent-server-conversation-service.types";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "./proxy";

/** Fetch the live workspace hooks reported for a Cloud conversation. */
export async function fetchCloudConversationHooks(
  conversationId: string,
  timeoutSeconds?: number,
): Promise<GetHooksResponse> {
  const backend = getActiveBackend().backend;
  if (backend.kind !== "cloud") {
    throw new Error("Cloud hooks call requires a cloud backend.");
  }

  return callCloudProxy<GetHooksResponse>({
    backend,
    method: "GET",
    path: `/api/v1/app-conversations/${encodeURIComponent(conversationId)}/hooks`,
    timeoutSeconds,
  });
}
