import { isAxiosError } from "axios";
import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { mapAnyGitStatusToClientStatus } from "#/utils/git-status-mapper";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import type {
  GitChange,
  GitChangeDiff,
  GitCommitsPage,
} from "../open-hands.types";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { isSdkHttpStatusError } from "../agent-server-compatibility";

interface AgentServerGitChange {
  status: string;
  path: string;
}

interface AgentServerGitCommit {
  sha: string;
  short_sha: string;
  subject: string;
  author: string;
  timestamp: string;
}

interface AgentServerGitCommitsPage {
  commits: AgentServerGitCommit[];
  has_more: boolean;
}

/**
 * Git operations for agent-server conversations.
 *
 * In **local** mode the runtime is reachable directly from the browser
 * (it's `127.0.0.1:18000`); the SDK's `RemoteWorkspace` calls land
 * fine. In **cloud** mode the runtime is at
 * `*.prod-runtime.all-hands.dev`, which doesn't allow CORS from
 * `localhost`. So cloud-mode calls hit the cloud API's
 * `GET /api/v1/app-conversations/{id}/git/{changes,diff}` proxy
 * endpoints instead — the server resolves the conversation's runtime
 * and makes the hop itself with the sandbox's session API key, and the
 * cloud API's CORS is permissive for bearer-token requests.
 */

/**
 * The cloud runtime's `/api/git/{changes,diff}` endpoints prepend
 * `/workspace/` to relative paths (so a relative arg like
 * `workspace/project` becomes `/workspace/workspace/project` and 404s).
 * `getGitPath` returns the local agent-server's relative convention by
 * default; normalize to an absolute path before sending to the cloud
 * runtime.
 */
function toAbsoluteRuntimePath(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}

/**
 * GET an arbitrary runtime git endpoint in both backend modes. Local mode
 * uses the SDK's public `HttpClient` (the typed `gitChanges`/`gitDiff`
 * wrappers don't know the newer endpoints/params, and they re-wrap errors,
 * dropping the `.status` needed for 404 feature-detection). Cloud mode goes
 * through the generic cloud-proxy envelope with the sandbox's session API
 * key — the same hop `executeCommand`/`downloadFile` use — so no dedicated
 * cloud API endpoints are required.
 */
async function getFromRuntime<T>(
  conversationUrl: string | null | undefined,
  sessionApiKey: string | null | undefined,
  apiPath: string,
  params: { path: string } & Record<string, string>,
): Promise<T> {
  const active = getActiveBackend().backend;

  if (active.kind === "cloud" && conversationUrl) {
    const search = new URLSearchParams({
      ...params,
      path: toAbsoluteRuntimePath(params.path),
    });
    return callCloudProxy<T>({
      backend: active,
      method: "GET",
      hostOverride: buildHttpBaseUrl(conversationUrl),
      path: `${apiPath}?${search.toString()}`,
      authMode: "session-api-key",
      sessionApiKey: sessionApiKey ?? undefined,
    });
  }

  const response = await new RemoteWorkspace(
    getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
  ).client.get<T>(apiPath, { params });
  return response.data;
}

/**
 * The commit endpoints are newer than some deployed agent servers. A 404
 * (route absent) means "server too old" — callers hide the commits section
 * instead of surfacing an error. Local calls and cloud-proxy calls (via the
 * shared TypeScript client) both throw the SDK's `HttpError`; the axios
 * check is kept as a safety net for axios-shaped errors.
 */
function isEndpointMissingError(error: unknown): boolean {
  return (
    isSdkHttpStatusError(error, 404) ||
    (isAxiosError(error) && error.response?.status === 404)
  );
}

class AgentServerGitService {
  static async getGitChanges(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
  ): Promise<GitChange[]> {
    const active = getActiveBackend().backend;

    if (active.kind === "cloud" && conversationId) {
      const params = new URLSearchParams();
      params.set("path", toAbsoluteRuntimePath(path));
      const data = await callCloudProxy<AgentServerGitChange[]>({
        backend: active,
        method: "GET",
        path: `/api/v1/app-conversations/${conversationId}/git/changes?${params.toString()}`,
      });
      if (!Array.isArray(data)) {
        throw new Error(
          "Invalid response from runtime - runtime may be unavailable",
        );
      }
      return data.map((change) => ({
        status: mapAnyGitStatusToClientStatus(
          String(change.status) as Parameters<
            typeof mapAnyGitStatusToClientStatus
          >[0],
        ),
        path: change.path,
      }));
    }

    // No `ref`: let the server auto-detect the base (origin branch /
    // merge-base), so changes the agent has already committed still show
    // up. `ref: "HEAD"` would go blank after every `git commit` — and the
    // cloud-proxy branch above already omits `ref`.
    const changes = await new RemoteWorkspace(
      getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
    ).gitChanges(path);

    if (!Array.isArray(changes)) {
      throw new Error(
        "Invalid response from runtime - runtime may be unavailable",
      );
    }

    return changes.map((change) => ({
      status: mapAnyGitStatusToClientStatus(
        String(change.status) as Parameters<
          typeof mapAnyGitStatusToClientStatus
        >[0],
      ),
      path: change.path,
    }));
  }

  /**
   * List the conversation's commits (display-base..HEAD, newest first).
   * Resolves to `null` when the agent server predates the endpoint (404),
   * so callers can hide the commits section instead of erroring.
   */
  static async getGitCommits(
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
    limit = 50,
  ): Promise<GitCommitsPage | null> {
    try {
      const page = await getFromRuntime<AgentServerGitCommitsPage>(
        conversationUrl,
        sessionApiKey,
        "/api/git/commits",
        { path, limit: String(limit) },
      );
      return {
        commits: (page?.commits ?? []).map((commit) => ({
          sha: commit.sha,
          shortSha: commit.short_sha,
          subject: commit.subject,
          author: commit.author,
          timestamp: commit.timestamp,
        })),
        hasMore: Boolean(page?.has_more),
      };
    } catch (error) {
      if (isEndpointMissingError(error)) return null;
      throw error;
    }
  }

  /** Files changed by a single commit (vs its first parent). */
  static async getCommitChanges(
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
    sha: string,
  ): Promise<GitChange[]> {
    const changes = await getFromRuntime<AgentServerGitChange[]>(
      conversationUrl,
      sessionApiKey,
      `/api/git/commits/${encodeURIComponent(sha)}/changes`,
      { path },
    );

    if (!Array.isArray(changes)) {
      throw new Error(
        "Invalid response from runtime - runtime may be unavailable",
      );
    }

    return changes.map((change) => ({
      status: mapAnyGitStatusToClientStatus(
        String(change.status) as Parameters<
          typeof mapAnyGitStatusToClientStatus
        >[0],
      ),
      path: change.path,
    }));
  }

  static async getGitChangeDiff(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
    commit?: string,
  ): Promise<GitChangeDiff> {
    if (commit) {
      // Per-commit diff: both sides come from git objects on the server,
      // so files the commit deleted still render.
      const commitDiff = await getFromRuntime<
        GitChangeDiff & { diff?: string }
      >(conversationUrl, sessionApiKey, "/api/git/diff", { path, commit });
      return {
        modified: commitDiff?.modified ?? "",
        original: commitDiff?.original ?? "",
        ...(commitDiff?.diff ? { diff: commitDiff.diff } : {}),
      } as GitChangeDiff;
    }

    const active = getActiveBackend().backend;

    if (active.kind === "cloud" && conversationId) {
      const params = new URLSearchParams();
      params.set("path", toAbsoluteRuntimePath(path));
      const diff = await callCloudProxy<GitChangeDiff & { diff?: string }>({
        backend: active,
        method: "GET",
        path: `/api/v1/app-conversations/${conversationId}/git/diff?${params.toString()}`,
      });
      return {
        modified: diff?.modified ?? "",
        original: diff?.original ?? "",
        ...(diff?.diff ? { diff: diff.diff } : {}),
      } as GitChangeDiff;
    }

    // No `ref` for the same reason as getGitChanges: the base must match
    // the one the change list was computed against.
    const diff = (await new RemoteWorkspace(
      getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
    ).gitDiff(path)) as GitChangeDiff & { diff?: string };

    return {
      modified: diff.modified ?? "",
      original: diff.original ?? "",
      ...(diff.diff ? { diff: diff.diff } : {}),
    } as GitChangeDiff;
  }
}

export default AgentServerGitService;
