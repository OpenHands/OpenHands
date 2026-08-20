// @spec WUP-001 — Resolve relative working dirs against /api/file/home
import { FileClient } from "@openhands/typescript-client/clients";
import {
  getAgentServerClientOptions,
  type AgentServerClientOverrides,
} from "./agent-server-client-options";

/**
 * Cache the agent-server's home directory per host so we only round-trip
 * `/api/file/home` once per backend. The home dir is effectively static for
 * the lifetime of a running agent-server (it's `Path.home()` on the host),
 * so caching is safe and avoids hammering the endpoint on every upload.
 *
 * The cache holds `Promise<string>` rather than `string` so concurrent
 * callers share a single in-flight request.
 */
const homeDirCache = new Map<string, Promise<string>>();

function isAbsolutePath(path: string): boolean {
  // Treat POSIX-style and Windows-style absolute paths as absolute.
  // The agent-server itself is POSIX on Linux/macOS and uses `\\` style
  // on Windows; we just need to know whether `Path(path).is_absolute()`
  // would return true on the server.
  //
  // Patterns covered:
  //   POSIX absolute:      /foo/bar
  //   Windows drive:       C:\foo  or  C:/foo
  //   Windows UNC:         \\server\share  (starts with `\`, matches `[/\\]`)
  return /^([/\\]|[a-zA-Z]:[/\\])/.test(path);
}

/**
 * Join a parent directory and a relative child segment with a forward slash,
 * collapsing any duplicate separators that result from the join.
 */
function joinPath(parent: string, child: string): string {
  const left = parent.replace(/[/\\]+$/, "");
  const right = child.replace(/^[/\\]+/, "");
  return `${left}/${right}`;
}

/**
 * Fetch and cache the agent-server's home directory via `GET /api/file/home`.
 *
 * The result is the absolute path returned by `Path.home()` on the
 * agent-server host (e.g. `/Users/foo`, `/root`, or `C:\\Users\\Foo`). This
 * is the most reliable absolute, writable anchor the agent-server API
 * currently exposes — `/server_info` doesn't include the process CWD.
 *
 * @param overrides Same shape as `getAgentServerClientOptions` — lets cloud
 *   sandboxes pass a `conversationUrl` + `sessionApiKey` so the lookup goes
 *   to the per-conversation runtime rather than the bundled local backend.
 */
export async function getAgentServerHomeDir(
  overrides: AgentServerClientOverrides = {},
): Promise<string> {
  const options = getAgentServerClientOptions(overrides);
  const cacheKey = options.host;
  const cached = homeDirCache.get(cacheKey);
  if (cached) return cached;

  const lookup = (async () => {
    const { home } = await new FileClient(options).getHome();
    if (!home || typeof home !== "string") {
      throw new Error("Agent server returned an empty home directory");
    }
    return home.replace(/[/\\]+$/, "");
  })();

  homeDirCache.set(cacheKey, lookup);
  try {
    return await lookup;
  } catch (error) {
    // Don't cache failures — let the next call retry.
    homeDirCache.delete(cacheKey);
    throw error;
  }
}

/** Test-only helper. */
export function clearAgentServerHomeDirCache(): void {
  homeDirCache.clear();
}

/**
 * Resolve a (possibly relative) working dir to an absolute path the
 * agent-server's file APIs will accept.
 *
 * - If `workingDir` is already absolute, returns it unchanged.
 * - Otherwise prepends the agent-server's home dir (looked up via
 *   `/api/file/home` and cached). This matches how the published binary
 *   and Docker entrypoint expect to anchor relative working dirs: under
 *   `~/workspace/project` rather than the filesystem root.
 *
 * Why this matters: the agent-server's `/api/file/upload` endpoint requires
 * an absolute path and `mkdir -p`s the parent. Naively prepending `/` to a
 * relative dir like `workspace/project/<hex>` produces `/workspace/...`,
 * which on macOS lives under the SIP-protected read-only root and fails
 * with `Errno 30: Read-only file system: '/workspace'`. Resolving against
 * `Path.home()` instead puts the path somewhere reliably writable.
 */
export async function resolveAbsoluteAgentServerPath(
  workingDir: string,
  overrides: AgentServerClientOverrides = {},
): Promise<string> {
  const trimmed = workingDir.replace(/[/\\]+$/, "");
  if (!trimmed) {
    return getAgentServerHomeDir(overrides);
  }
  if (isAbsolutePath(trimmed)) return trimmed;

  const home = await getAgentServerHomeDir(overrides);
  return joinPath(home, trimmed);
}
