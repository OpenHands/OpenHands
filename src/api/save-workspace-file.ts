import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  getSafeUploadFileName,
  resolveAbsoluteWorkspacePath,
  resolveConversationUploadWorkingDir,
} from "#/api/workspace-upload-path";
import { resolveConversationRuntime } from "#/api/conversation-file-upload.api";

/**
 * Join a workspace-relative path onto an absolute working dir, rejecting
 * empty / traversal segments so a UI-selected path cannot escape the
 * conversation workspace.
 */
export function joinWorkspaceRelativePath(
  absoluteDir: string,
  relativePath: string,
): string {
  const safeRelative = relativePath
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part && part !== "." && part !== "..")
    .join("/");

  if (!safeRelative) {
    throw new Error("Invalid file path");
  }

  return `${absoluteDir.replace(/[/\\]+$/, "")}/${safeRelative}`;
}

/**
 * Overwrite a text file in the active conversation workspace.
 *
 * Uses the same {@link RemoteWorkspace.uploadText} / `/api/file/upload`
 * path as chat attachments, but targets the existing relative path
 * instead of dumping into the workspace root by basename.
 */
export async function saveWorkspaceFile(options: {
  conversation: AppConversation;
  relativePath: string;
  content: string;
}): Promise<void> {
  const { conversation, relativePath, content } = options;
  const workingDir = await resolveConversationUploadWorkingDir(
    conversation.id,
    conversation,
  );
  const runtime = await resolveConversationRuntime(
    conversation.id,
    conversation,
  );
  const isCloud = getActiveBackend().backend.kind === "cloud";

  const conversationUrl =
    conversation.conversation_url?.trim() || runtime.conversationUrl;
  const sessionApiKey =
    conversation.session_api_key?.trim() || runtime.sessionApiKey;

  if (isCloud && (!conversationUrl || !sessionApiKey)) {
    throw new Error(
      "Conversation sandbox is still starting. Wait for it to finish, then try again.",
    );
  }

  const absoluteDir = await resolveAbsoluteWorkspacePath(workingDir, {
    conversationUrl,
    sessionApiKey,
  });
  const destinationPath = joinWorkspaceRelativePath(absoluteDir, relativePath);
  const fileName = getSafeUploadFileName(relativePath);

  const workspace = new RemoteWorkspace(
    getAgentServerClientOptions({
      conversationUrl,
      sessionApiKey,
      workingDir,
    }),
  );

  const result = await workspace.uploadText(content, destinationPath, fileName);

  if (result.success === false) {
    throw new Error(result.error || "Failed to save file");
  }
}
