import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { useQuery } from "@tanstack/react-query";

import { getActiveBackend } from "#/api/backend-registry/active-store";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";

export interface WorkspaceSession {
  /**
   * Absolute URL prefix for the conversation's static workspace fileserver,
   * always ending in a `/`. Append a relative path to address a single file
   * (e.g. `${baseUrl}index.html`).
   */
  baseUrl: string;
}

/**
 * Mint a workspace static-asset session for the active conversation.
 *
 * Calling `POST /api/auth/workspace-session` exchanges the conversation's
 * `X-Session-API-Key` for an `oh_workspace_session_key` cookie scoped to
 * `/api/conversations`. Once that cookie is set the browser can embed
 * workspace artifacts directly as `<iframe src>` / `<img src>` / top-level
 * navigations — which it cannot do when the only credential is a custom
 * request header.
 *
 * Local-only. The cookie flow is useless on cloud: the POST would be a
 * server-side hop through `callCloudProxy` (so the `Set-Cookie` never
 * reaches the browser jar), and the runtime sandbox is cross-origin with
 * the GUI (so `fetch(staticUrl, { credentials: "include" })` couldn't
 * attach the cookie anyway). On cloud, `useWorkspaceFileContent` fetches
 * bytes through `callCloudProxy` directly and renders binary kinds as
 * `data:` URIs, so this hook stays disabled and returns `data: null`.
 *
 * We treat the call as cache-once-per-conversation: the cookie lives in
 * the browser jar, so re-issuing the POST on every component remount is
 * wasted work. `staleTime: Infinity` keeps the cached `baseUrl` in place
 * for the lifetime of the conversation; only switching conversations (a
 * different `conversationId` in the query key) re-runs it.
 *
 * Returns `null` from `data` until the session has been minted, so
 * callers can gate iframe / img rendering on a definite "the cookie is
 * set" signal rather than guessing.
 */
export function useWorkspaceSession(): {
  data: WorkspaceSession | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
} {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();

  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const isLocal = getActiveBackend().backend.kind === "local";

  const enabled = runtimeIsReady && !!conversationId && isLocal;

  const query = useQuery<WorkspaceSession>({
    queryKey: [
      "workspace-session",
      conversationId,
      conversationUrl,
      sessionApiKey,
    ],
    queryFn: async () => {
      const workspace = new RemoteWorkspace(
        getAgentServerClientOptions({
          conversationUrl,
          sessionApiKey,
        }),
      );
      const baseUrl = await workspace.startWorkspaceSession(conversationId!);
      return { baseUrl };
    },
    enabled,
    // The cookie is sticky; minting it once per page-load is plenty.
    staleTime: Infinity,
    gcTime: Infinity,
    // No auto-retry: a 401 here means the session API key isn't valid for
    // this conversation, which won't fix itself on a second POST. The
    // user can re-trigger by reloading or switching conversation.
    retry: false,
  });

  return {
    data: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
    error: (query.error as Error | null) ?? null,
  };
}

/**
 * Append a workspace-relative path to a base URL produced by
 * {@link useWorkspaceSession}, URL-encoding each segment but preserving
 * `/` separators. Pass an empty / undefined `relativePath` to get the
 * directory base back (server then falls back to its `index.html`).
 */
export function joinWorkspaceUrl(
  baseUrl: string,
  relativePath?: string | null,
): string {
  const cleaned = (relativePath ?? "").replace(/^\/+/, "");
  if (!cleaned) return baseUrl;
  const encoded = cleaned
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  // baseUrl is guaranteed to end with a single trailing slash by the
  // server's contract (startWorkspaceSession returns `${host}/api/.../workspace/`).
  return `${baseUrl}${encoded}`;
}
