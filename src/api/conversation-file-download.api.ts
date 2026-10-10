import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { resolveAbsoluteAgentServerPath } from "#/api/agent-server-home";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { resolveConversationRuntime } from "#/api/conversation-file-upload.api";
import { downloadRuntimeFile } from "#/api/runtime-service/file-download";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

// @spec FD-001 — Download original bytes from the conversation's runtime
export async function downloadConversationFile(
  conversation: AppConversation,
  path: string,
): Promise<Blob> {
  const workingDir = conversation.workspace?.working_dir;
  if (!conversation.id || !workingDir || !path) {
    throw new Error("Conversation workspace is unavailable");
  }
  const isCloud = getActiveBackend().backend.kind === "cloud";
  // Snapshot the local backend before resolving anything asynchronously.
  const localOptions = isCloud
    ? null
    : getAgentServerClientOptions({
        conversationId: conversation.id,
        conversationUrl: conversation.conversation_url,
        sessionApiKey: conversation.session_api_key,
        workingDir,
      });
  const runtime = await resolveConversationRuntime(
    conversation.id,
    conversation,
  );
  if (isCloud && (!runtime.conversationUrl || !runtime.sessionApiKey)) {
    throw new Error("Conversation runtime is unavailable");
  }
  const overrides = {
    conversationId: conversation.id,
    conversationUrl: runtime.conversationUrl,
    sessionApiKey: runtime.sessionApiKey,
  };
  const options = localOptions ?? getAgentServerClientOptions(overrides);
  const absolutePath = await resolveAbsoluteAgentServerPath(
    `${workingDir.replace(/[/\\]+$/, "")}/${path}`,
    { ...overrides, host: options.host, apiKey: options.apiKey },
  );
  const bytes = await downloadRuntimeFile(options, absolutePath);
  return new Blob([bytes], { type: "application/octet-stream" });
}
