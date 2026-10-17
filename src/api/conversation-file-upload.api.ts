import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { batchGetCloudConversations } from "#/api/cloud/conversation-service.api";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { FileUploadSuccessResponse } from "#/api/open-hands.types";
import {
  buildWorkspaceUploadPath,
  getSafeUploadFileName,
  resolveConversationUploadWorkingDir,
} from "#/api/workspace-upload-path";

const FILE_UPLOAD_CONCURRENCY = 5;

export interface ConversationRuntimeContext {
  conversationUrl: string | null;
  sessionApiKey: string | null;
  workingDir?: string | null;
}

/**
 * Resolve the sandbox runtime URL + session key needed for file upload and
 * send-event calls. Cloud conversations only exist on the provisioned runtime,
 * not on the bundled local agent-server.
 */
export async function resolveConversationRuntime(
  conversationId: string,
  currentConversation?: AppConversation | null,
): Promise<ConversationRuntimeContext> {
  const activeConversation =
    currentConversation?.id === conversationId
      ? currentConversation
      : ConversationService.getCurrentConversation()?.id === conversationId
        ? ConversationService.getCurrentConversation()
        : null;

  if (
    activeConversation?.conversation_url?.trim() &&
    activeConversation.session_api_key?.trim()
  ) {
    return {
      conversationUrl: activeConversation.conversation_url.trim(),
      sessionApiKey: activeConversation.session_api_key.trim(),
      workingDir: activeConversation.workspace?.working_dir?.trim() ?? null,
    };
  }

  if (getActiveBackend().backend.kind === "cloud") {
    const [conversation] = await batchGetCloudConversations([conversationId]);
    return {
      conversationUrl: conversation?.conversation_url?.trim() ?? null,
      sessionApiKey: conversation?.session_api_key?.trim() ?? null,
      workingDir: conversation?.workspace?.working_dir?.trim() ?? null,
    };
  }

  return {
    conversationUrl: null,
    sessionApiKey: null,
    workingDir: activeConversation?.workspace?.working_dir?.trim() ?? null,
  };
}

function requireCloudRuntime(
  runtime: ConversationRuntimeContext,
): ConversationRuntimeContext & {
  conversationUrl: string;
  sessionApiKey: string;
} {
  if (!runtime.conversationUrl || !runtime.sessionApiKey) {
    throw new Error(
      "Conversation sandbox is still starting. Wait for it to finish, then try again.",
    );
  }
  return {
    conversationUrl: runtime.conversationUrl,
    sessionApiKey: runtime.sessionApiKey,
  };
}

/**
 * Upload attachments into the conversation workspace. Local conversations use
 * the bundled agent-server; cloud conversations target the provisioned runtime.
 */
export async function uploadFilesToConversation(
  conversationId: string,
  files: File[],
  currentConversation?: AppConversation | null,
): Promise<FileUploadSuccessResponse> {
  try {
    const activeConversation =
      currentConversation?.id === conversationId
        ? currentConversation
        : ConversationService.getCurrentConversation()?.id === conversationId
          ? ConversationService.getCurrentConversation()
          : null;

    const runtime = await resolveConversationRuntime(
      conversationId,
      activeConversation,
    );
    const isCloud = getActiveBackend().backend.kind === "cloud";

    const sessionApiKey =
      activeConversation?.session_api_key ?? runtime.sessionApiKey;
    const conversationUrl =
      activeConversation?.conversation_url ?? runtime.conversationUrl;

    const workingDir =
      runtime.workingDir?.trim() ||
      (await resolveConversationUploadWorkingDir(
        conversationId,
        activeConversation,
      ));

    if (isCloud) {
      const cloudRuntime = requireCloudRuntime({
        conversationUrl,
        sessionApiKey,
      });
      return await uploadFilesToRuntime({
        conversationId,
        files,
        workingDir,
        conversationUrl: cloudRuntime.conversationUrl,
        sessionApiKey: cloudRuntime.sessionApiKey,
      });
    }

    return await uploadFilesToRuntime({
      conversationId,
      files,
      workingDir,
      conversationUrl,
      sessionApiKey,
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : "Upload failed";
    return {
      uploaded_files: [],
      skipped_files: files.map((file) => ({
        name: file.name,
        reason,
      })),
    };
  }
}

async function uploadFilesToRuntime(options: {
  conversationId: string;
  files: File[];
  workingDir: string;
  conversationUrl: string | null;
  sessionApiKey: string | null;
}): Promise<FileUploadSuccessResponse> {
  const { files, workingDir, conversationUrl, sessionApiKey } = options;
  const workspace = new RemoteWorkspace(
    getAgentServerClientOptions({
      conversationId: options.conversationId,
      conversationUrl,
      sessionApiKey,
      workingDir,
    }),
  );

  const uploadFile = async (file: File) => {
    try {
      const safeName = getSafeUploadFileName(file.name);
      // @spec WUP-001 — Build an absolute upload path that's anchored against
      // the agent-server's home dir (when `workingDir` is relative) instead
      // of the filesystem root. Without this, default conversations whose
      // working_dir is `workspace/project/<hex>` (relative) land at
      // `/workspace/project/<hex>/...` on the agent-server, which on macOS
      // and fresh containers is a read-only mount.
      const uploadPath = await buildWorkspaceUploadPath(file.name, workingDir, {
        conversationUrl,
        sessionApiKey,
      });
      await workspace.fileUpload(file, uploadPath);
      return { uploadedFile: safeName, skippedFile: null };
    } catch (error) {
      return {
        uploadedFile: null,
        skippedFile: {
          name: file.name,
          reason: error instanceof Error ? error.message : "Upload failed",
        },
      };
    }
  };

  const results: Awaited<ReturnType<typeof uploadFile>>[] = [];
  for (let index = 0; index < files.length; index += FILE_UPLOAD_CONCURRENCY) {
    const batch = files.slice(index, index + FILE_UPLOAD_CONCURRENCY);
    results.push(...(await Promise.all(batch.map(uploadFile))));
  }

  return {
    uploaded_files: results.flatMap((result) =>
      result.uploadedFile ? [result.uploadedFile] : [],
    ),
    skipped_files: results.flatMap((result) =>
      result.skippedFile ? [result.skippedFile] : [],
    ),
  };
}
