import { useSyncExternalStore } from "react";
import { useQuery } from "@tanstack/react-query";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { listCloudConversationFiles } from "#/api/cloud/conversation-service.api";
import {
  getSnapshot,
  subscribeActiveBackend,
} from "#/api/backend-registry/active-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";
import { useWorkspaceFileDiscovery } from "./use-workspace-file-discovery";
import {
  buildWorkspaceFileListCommand,
  parseWorkspaceFileList,
} from "#/utils/workspace-file-discovery";

// Cap the number of files we render so a giant repo doesn't freeze the UI.
const MAX_FILES = 2000;

export interface WorkspaceFilesResult {
  data: string[] | undefined;
  isLoading: boolean;
  isTruncated?: boolean;
}

function normalizePath(path: string): string {
  // Strip a leading "./" so paths render cleanly in the UI.
  return path.startsWith("./") ? path.slice(2) : path;
}

/**
 * Local-backend listing: enumerate every regular file beneath the active
 * conversation's working directory via `find` over the agent-server's
 * `/api/bash/execute_bash_command`, excluding common heavy/build directories.
 * Returns paths relative to the working dir (e.g. `src/index.html`).
 *
 * Local only: the browser can't drive `/api/bash/execute_bash_command` on a
 * cloud runtime (no CORS, and the `/api/cloud-proxy` hop was removed). Cloud
 * backends hit a first-class cloud API listing endpoint instead — see
 * `useCloudWorkspaceFiles`.
 */
function useLocalWorkspaceFiles(enabled: boolean): WorkspaceFilesResult {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady({ allowAgentError: true });

  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const workingDir = conversation?.workspace?.working_dir?.trim();
  const discovery = useWorkspaceFileDiscovery(enabled ? workingDir : undefined);
  const { options } = discovery;

  const query = useQuery({
    queryKey: [
      "workspace-files",
      conversationId,
      conversationUrl,
      sessionApiKey,
      workingDir,
      options,
    ],
    queryFn: async () => {
      const result = await AgentServerRuntimeService.executeCommand(
        conversationUrl,
        sessionApiKey,
        buildWorkspaceFileListCommand(options),
        workingDir,
        30,
        conversationId,
      );

      if (result.exit_code !== 0) {
        throw new Error(
          result.stderr?.trim() || "Failed to list workspace files",
        );
      }

      return parseWorkspaceFileList(result.stdout, options.maxFiles);
    },
    enabled:
      enabled &&
      runtimeIsReady &&
      !!conversationId &&
      !!workingDir &&
      !discovery.isLoading,
    retry: false,
    staleTime: 1000 * 30,
    gcTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });

  return {
    data: query.data?.paths,
    isLoading: discovery.isLoading || query.isLoading,
    isTruncated: query.data?.isTruncated ?? false,
  };
}

/**
 * Cloud-backend listing: enumerate the full workspace tree via the cloud API's
 * first-class `GET /api/v1/app-conversations/{id}/files` endpoint, which
 * resolves the conversation's runtime and runs the same bounded `find`
 * server-side (see enterprise `list_conversation_files`). This is the same
 * server-side runtime-hop transport the git-changes/diff and single-file read
 * already use, so it works without CORS or the removed `/api/cloud-proxy` hop.
 *
 * Unlike the previous git-changes-based approach, this returns unchanged
 * tracked files too, so a conversation attached to a large existing repo shows
 * the whole tree — matching the local-backend experience. Paths come back
 * relative to the working dir (e.g. `src/index.html`).
 */
function useCloudWorkspaceFiles(enabled: boolean): WorkspaceFilesResult {
  // Source the id from the route (like the diff/commits cloud hooks), NOT from
  // `useActiveConversation().data.id`: the cloud API listing call only needs
  // the id, and gating on the batch-get query's data would keep the query
  // disabled — and never fire — whenever that data is null or still loading.
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady({ allowAgentError: true });

  const selectedRepository = conversation?.selected_repository;
  const workingDir = conversation?.workspace?.working_dir?.trim();

  // Anchor against an absolute workspace path the same way the diff view and
  // single-file read do (`getGitPath` defaults to a relative convention; the
  // cloud runtime needs an absolute path).
  const gitPath = getGitPath(selectedRepository, workingDir);
  const absolutePath = gitPath.startsWith("/") ? gitPath : `/${gitPath}`;

  const query = useQuery<string[]>({
    queryKey: ["workspace-files-cloud", conversationId, absolutePath],
    queryFn: async () => {
      const files = await listCloudConversationFiles(
        conversationId!,
        absolutePath,
      );
      const normalized = files.map(normalizePath).filter(Boolean);
      return Array.from(new Set(normalized)).slice(0, MAX_FILES);
    },
    enabled: enabled && runtimeIsReady && !!conversationId,
    retry: false,
    staleTime: 1000 * 30,
    gcTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });

  return { data: query.data, isLoading: query.isLoading };
}

/**
 * Lists the files shown in the Files tab for the active conversation.
 *
 * Both backends enumerate the full workspace tree. Local backends run bash
 * `find` directly against the agent-server; cloud backends call the cloud
 * API's first-class file-listing endpoint, which runs the same `find`
 * server-side on the conversation's runtime (see `useCloudWorkspaceFiles`).
 *
 * Cloud detection reads the backend-registry store (via `useSyncExternalStore`)
 * rather than the `ActiveBackendProvider` context. The transport layer that
 * actually issues the requests — `executeCommand`, `getGitChanges`, the cloud
 * file-read — all branch on the *store* (`getActiveBackend()`), so the Files
 * tab must use the same source. Reading the context here can disagree with the
 * store (its `useActiveBackend` fallback synthesizes a *local* backend when the
 * provider isn't in scope), which would run the local bash path against a cloud
 * backend: `executeCommand` then POSTs to the removed `/api/cloud-proxy` (405)
 * and the cloud `/files` call never fires.
 */
export function useWorkspaceFiles(): WorkspaceFilesResult {
  const snapshot = useSyncExternalStore(
    subscribeActiveBackend,
    getSnapshot,
    getSnapshot,
  );
  const isCloud = snapshot.active.backend.kind === "cloud";

  const local = useLocalWorkspaceFiles(!isCloud);
  const cloud = useCloudWorkspaceFiles(isCloud);

  return isCloud ? cloud : local;
}
