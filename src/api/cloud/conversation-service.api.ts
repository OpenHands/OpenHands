import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { getStoredConversationMetadata } from "../conversation-metadata-store";
import type {
  AppConversation,
  AppConversationPage,
  AppConversationStartRequest,
  AppConversationStartTask,
} from "../conversation-service/agent-server-conversation-service.types";
import { AGENT_CANVAS_CLIENT_HEADERS } from "../client-source";
import { callCloudProxy } from "./proxy";

/**
 * The cloud backend does not always echo `selected_repository` /
 * `selected_branch` / `git_provider` back from
 * `GET /api/v1/app-conversations` until its own background hydration
 * completes. We persist the selection to local storage at connect time
 * (see `AgentServerConversationService.updateConversationRepository`)
 * and overlay it here so the chat-page git control bar reflects the
 * connection immediately, instead of snapping back to the empty
 * "Connect Repo" state on every refetch.
 *
 * Server values take precedence whenever they're populated; the
 * local-storage fallback only fills in fields the server returned as
 * `null`/`undefined`.
 */
function overlayStoredRepoSelection(
  conversation: AppConversation | null,
): AppConversation | null {
  if (!conversation?.id) return conversation;
  const stored = getStoredConversationMetadata(conversation.id);
  if (!stored) return conversation;

  return {
    ...conversation,
    selected_repository:
      conversation.selected_repository ?? stored.selected_repository ?? null,
    selected_branch:
      conversation.selected_branch ?? stored.selected_branch ?? null,
    git_provider: conversation.git_provider ?? stored.git_provider ?? null,
    selected_workspace:
      conversation.selected_workspace ?? stored.selected_workspace ?? null,
  };
}

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud conversations call requires a cloud backend.");
  }
  return active;
}

/**
 * Search the cloud app-conversations list. Mirrors the local
 * `AgentServerConversationService.searchConversations` interface but calls
 * the cloud endpoint `/api/v1/app-conversations/search`.
 */
export async function searchCloudConversations(
  limit: number = 20,
  pageId?: string,
): Promise<AppConversationPage> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (pageId) params.set("page_id", pageId);
  params.set("sort_order", "UPDATED_AT_DESC");

  const data = await callCloudProxy<{
    items: AppConversation[];
    next_page_id: string | null;
  }>({
    backend,
    method: "GET",
    path: `/api/v1/app-conversations/search?${params.toString()}`,
  });

  return {
    items: (data?.items ?? []).map(
      (item) => overlayStoredRepoSelection(item) as AppConversation,
    ),
    next_page_id: data?.next_page_id ?? null,
  };
}

/**
 * Batch-fetch cloud app-conversations by id. Mirrors the local
 * `AgentServerConversationService.batchGetAppConversations` interface.
 */
export async function batchGetCloudConversations(
  ids: string[],
): Promise<(AppConversation | null)[]> {
  if (ids.length === 0) return [];
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  for (const id of ids) params.append("ids", id);
  const data = await callCloudProxy<(AppConversation | null)[]>({
    backend,
    method: "GET",
    path: `/api/v1/app-conversations?${params.toString()}`,
  });
  return (data ?? []).map(overlayStoredRepoSelection);
}

/**
 * Create a v1 app-conversation on the cloud backend.
 *
 * Mirrors OpenHands' cloud flow: POST /api/v1/app-conversations with the
 * `AppConversationStartRequest` payload, returning a
 * `AppConversationStartTask`. The task is initially WORKING; the caller
 * polls `getCloudAppConversationStartTask` (3s cadence per OpenHands)
 * until status is READY (then `app_conversation_id`, `agent_server_url`,
 * and `session_api_key` are populated) or ERROR.
 *
 * This path does NOT use encrypted-settings round-tripping. Secrets stay
 * server-side on the cloud backend — the only auth carried is the cloud bearer
 * token, and the conversation runtime is
 * provisioned with its own ephemeral session_api_key returned in the
 * task.
 */
export async function createCloudAppConversation(
  request: AppConversationStartRequest,
): Promise<AppConversationStartTask> {
  const backend = getActiveCloudBackend();
  const data = await callCloudProxy<AppConversationStartTask>({
    backend,
    method: "POST",
    path: "/api/v1/app-conversations",
    body: request as unknown as Record<string, unknown>,
    headers: AGENT_CANVAS_CLIENT_HEADERS,
  });
  return data;
}

/**
 * Download a v1 app-conversation as a ZIP from the cloud backend. Mirrors
 * the local `AgentServerConversationService.downloadConversation` interface but
 * calls
 * `GET /api/v1/app-conversations/{id}/download`, which returns
 * `application/zip` with `Content-Disposition` set by the cloud backend.
 */
export async function downloadCloudConversation(
  conversationId: string,
): Promise<Blob> {
  const backend = getActiveCloudBackend();
  return callCloudProxy<Blob>({
    backend,
    method: "GET",
    path: `/api/v1/app-conversations/${conversationId}/download`,
    responseType: "blob",
  });
}

/**
 * Delete a v1 app-conversation on the cloud backend. Mirrors the local
 * `AgentServerConversationService.deleteConversation` interface but calls
 * `DELETE /api/v1/app-conversations/{id}`, which returns a JSON
 * `Success` envelope (discarded here — the caller only needs to know
 * the request didn't error).
 */
export async function deleteCloudConversation(
  conversationId: string,
): Promise<void> {
  const backend = getActiveCloudBackend();
  await callCloudProxy<unknown>({
    backend,
    method: "DELETE",
    path: `/api/v1/app-conversations/${conversationId}`,
  });
}

/**
 * Toggle the public-sharing flag on a cloud v1 app-conversation. Mirrors
 * OpenHands' `AgentServerConversationService.updateConversationPublicFlag`:
 * `PATCH /api/v1/app-conversations/{id}` with `{ public }`, returning
 * the updated conversation.
 */
export async function updateCloudConversationPublicFlag(
  conversationId: string,
  isPublic: boolean,
): Promise<AppConversation> {
  const backend = getActiveCloudBackend();
  const data = await callCloudProxy<AppConversation>({
    backend,
    method: "PATCH",
    path: `/api/v1/app-conversations/${conversationId}`,
    body: { public: isPublic },
  });
  return data;
}

/**
 * Pause the cloud sandbox backing a v1 app-conversation. Mirrors
 * OpenHands' `SandboxService.pauseSandbox`:
 * `POST /api/v1/sandboxes/{sandboxId}/pause` on the cloud backend, which stops
 * the runtime owning the conversation.
 */
export async function pauseCloudSandbox(sandboxId: string): Promise<void> {
  const backend = getActiveCloudBackend();
  await callCloudProxy<unknown>({
    backend,
    method: "POST",
    path: `/api/v1/sandboxes/${sandboxId}/pause`,
  });
}

/**
 * Resume a paused cloud sandbox. Mirrors OpenHands' `SandboxService.resumeSandbox`
 * by calling `POST /api/v1/sandboxes/{sandboxId}/resume` on the SaaS.
 *
 * This is the correct endpoint for waking a PAUSED sandbox. It is a
 * lightweight unpause — NOT the same as creating a new start task via
 * `POST /api/v1/app-conversations`, which provisions a fresh conversation
 * and is subject to the 120-second sandbox-start timeout.
 */
export async function resumeCloudSandbox(sandboxId: string): Promise<void> {
  const backend = getActiveCloudBackend();
  await callCloudProxy<unknown>({
    backend,
    method: "POST",
    path: `/api/v1/sandboxes/${sandboxId}/resume`,
  });
}

/**
 * Read a file from a cloud conversation's sandbox workspace. Mirrors
 * OpenHands' `AgentServerConversationService.readConversationFile` — hits
 * `GET /api/v1/app-conversations/{id}/file?file_path=...` on the cloud backend
 * and returns the file content as a string.
 */
export async function readCloudConversationFile(
  conversationId: string,
  filePath: string,
): Promise<string> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.append("file_path", filePath);
  const data = await callCloudProxy<string>({
    backend,
    method: "GET",
    path: `/api/v1/app-conversations/${conversationId}/file?${params.toString()}`,
  });
  return data ?? "";
}

/**
 * Fetch a single v1 app-conversation start task. Mirrors OpenHands'
 * `AgentServerConversationService.getStartTask` — uses the batch search endpoint
 * with a single id and unwraps the first result.
 */
export async function getCloudAppConversationStartTask(
  taskId: string,
): Promise<AppConversationStartTask | null> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("ids", taskId);
  const data = await callCloudProxy<(AppConversationStartTask | null)[]>({
    backend,
    method: "GET",
    path: `/api/v1/app-conversations/start-tasks?${params.toString()}`,
  });
  return data?.[0] ?? null;
}
