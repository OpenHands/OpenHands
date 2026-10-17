import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { ConversationClient } from "@openhands/typescript-client/clients";
import { HttpError } from "@openhands/typescript-client";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { isAbsolutePath } from "#/api/agent-server-home";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { batchGetCloudConversations } from "#/api/cloud/conversation-service.api";
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
  if (
    currentConversation?.id === conversationId &&
    currentConversation.conversation_url?.trim() &&
    currentConversation.session_api_key?.trim()
  ) {
    return {
      conversationUrl: currentConversation.conversation_url.trim(),
      sessionApiKey: currentConversation.session_api_key.trim(),
    };
  }

  if (getActiveBackend().backend.kind === "cloud") {
    const [conversation] = await batchGetCloudConversations([conversationId]);
    return {
      conversationUrl: conversation?.conversation_url?.trim() ?? null,
      sessionApiKey: conversation?.session_api_key?.trim() ?? null,
    };
  }

  return { conversationUrl: null, sessionApiKey: null };
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
 * Read the conversation's own `workspace.working_dir` from the runtime the
 * upload targets.
 *
 * The agent-server's `/api/file/upload` guard requires the destination to
 * resolve inside the working dir it stored for the conversation. Canvas can
 * only *guess* that path locally — a relative default (`workspace/project/<hex>`)
 * or a Cloud assumption (`/workspace/project`) that diverges from the dir the
 * runtime actually reports — and the guess then trips the guard with a 422
 * `path must be inside the conversation workspace`. Reading the runtime's
 * reported value makes the destination authoritative instead of guessed.
 *
 * Only an **absolute** runtime value is authoritative. The guard resolves a
 * relative stored dir against the server's own process directory, which Canvas
 * cannot know (it anchors relative dirs against `/api/file/home`), so adopting
 * a relative runtime value would point the upload somewhere the guard rejects
 * and would override an already-correct absolute local value.
 *
 * Returns `null` (never throws) when the runtime is unreachable, reports no
 * working dir, or reports a relative one, so the caller can fall back to the
 * local resolution.
 */
async function readRuntimeWorkingDir(
  runtime: ConversationRuntimeContext,
  conversationId: string,
): Promise<string | null> {
  try {
    const info = await new ConversationClient(
      getAgentServerClientOptions({
        conversationId,
        conversationUrl: runtime.conversationUrl,
        sessionApiKey: runtime.sessionApiKey,
      }),
    ).getConversation<{ workspace?: { working_dir?: string | null } | null }>(
      conversationId,
    );
    const dir = info?.workspace?.working_dir?.trim();
    // A relative dir is resolved by the guard against the server's own CWD,
    // which Canvas cannot know. Only an absolute value is authoritative;
    // otherwise keep the local resolution (WUP-001).
    return dir && isAbsolutePath(dir) ? dir : null;
  } catch {
    return null;
  }
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
  const runtime = await resolveConversationRuntime(
    conversationId,
    currentConversation,
  );
  const isCloud = getActiveBackend().backend.kind === "cloud";

  const sessionApiKey =
    currentConversation?.id === conversationId
      ? (currentConversation.session_api_key ?? runtime.sessionApiKey)
      : runtime.sessionApiKey;
  const conversationUrl =
    currentConversation?.id === conversationId
      ? (currentConversation.conversation_url ?? runtime.conversationUrl)
      : runtime.conversationUrl;

  // Cloud conversations must be fully provisioned before any runtime read or
  // upload. Validate before the (slower) working-dir read so a not-yet-ready
  // sandbox still surfaces the clear "still starting" message, and carry the
  // validated URL/key forward so the rest of the flow uses the same values.
  const runtimeCtx = isCloud
    ? requireCloudRuntime({ conversationUrl, sessionApiKey })
    : { conversationUrl, sessionApiKey };

  // @spec WUP-002 — the agent-server's upload guard resolves the destination
  // against the working dir *it* stored for the conversation. When the runtime
  // reports an absolute value, prefer it over the locally-derived one (a
  // relative default or an assumed Cloud path) which can fall outside the
  // workspace and trip the 422 guard. Otherwise — no runtime value, or a
  // relative one Canvas cannot resolve identically to the guard — keep the
  // local resolution, which preserves an already-correct absolute conversation
  // workspace (lazily evaluated, so its own network read is skipped when the
  // runtime already answered with an absolute value).
  const workingDir =
    (await readRuntimeWorkingDir(
      {
        conversationUrl: runtimeCtx.conversationUrl,
        sessionApiKey: runtimeCtx.sessionApiKey,
      },
      conversationId,
    )) ??
    (await resolveConversationUploadWorkingDir(
      conversationId,
      currentConversation,
    ));

  return uploadFilesToRuntime({
    conversationId,
    files,
    workingDir,
    conversationUrl: runtimeCtx.conversationUrl,
    sessionApiKey: runtimeCtx.sessionApiKey,
  });
}

/**
 * The agent-server's runtime guard rejects any `path` that does not resolve
 * inside the conversation's stored `workspace.working_dir` (see
 * `bind_local_conversation_runtime` in the SDK's `runtime_router.py`).
 */
const WORKSPACE_GUARD_DETAIL = "must be inside the conversation workspace";

/**
 * Map a raw upload failure to a message we can show the user. The workspace
 * guard's `422` body is an implementation detail; surface a human-readable
 * reason instead of the JSON payload.
 */
function friendlyUploadError(error: unknown): string {
  if (error instanceof HttpError && error.status === 422) {
    const detail = (error.response as { detail?: unknown } | null)?.detail;
    if (typeof detail === "string" && detail.includes(WORKSPACE_GUARD_DETAIL)) {
      return i18n.t(I18nKey.FILE_SERVICE$UPLOAD_WORKSPACE_MISMATCH);
    }
  }
  return error instanceof Error ? error.message : "Upload failed";
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
          reason: friendlyUploadError(error),
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
