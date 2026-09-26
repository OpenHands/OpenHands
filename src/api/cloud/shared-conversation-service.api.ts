import type { SharedConversation } from "@openhands/typescript-client";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

export interface SharedEventPage {
  items: OpenHandsEvent[];
  next_page_id: string | null;
}

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Shared conversation calls require a cloud backend.");
  }
  return active;
}

/**
 * Fetch a conversation shared with the caller from the cloud backend via
 * `GET /api/shared-conversations?ids=<id>`. The endpoint serves public
 * conversations to anyone and automation conversations to authenticated
 * members of their org; it returns `null` for anything else.
 */
export async function getCloudSharedConversation(
  conversationId: string,
): Promise<SharedConversation | null> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams({ ids: conversationId });
  const data = await callCloudProxy<(SharedConversation | null)[]>({
    backend,
    method: "GET",
    path: `/api/shared-conversations?${params.toString()}`,
  });
  return data?.[0] ?? null;
}

/**
 * Page through the events of a shared conversation on the cloud backend via
 * `GET /api/shared-events/search`.
 */
export async function searchCloudSharedEvents(options: {
  conversationId: string;
  limit?: number;
  pageId?: string;
}): Promise<SharedEventPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams({
    conversation_id: options.conversationId,
    limit: String(options.limit ?? 100),
  });
  if (options.pageId) params.set("page_id", options.pageId);
  return callCloudProxy<SharedEventPage>({
    backend,
    method: "GET",
    path: `/api/shared-events/search?${params.toString()}`,
  });
}
