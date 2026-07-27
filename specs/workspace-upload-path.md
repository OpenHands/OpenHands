# Workspace Upload Path Specs

---

### WUP-001: Relative working dirs are resolved against `/api/file/home`, not the filesystem root
- [x] When the frontend creates a conversation and the resolved working dir is **relative** (e.g. the `DEFAULT_WORKING_DIR = "workspace/project"` fallback), `AgentServerConversationService.createConversation` shall resolve it to an **absolute** path before sending `workspace.working_dir` to the agent-server, by prefixing the agent-server's home directory as returned by `GET /api/file/home`.
- [x] When the frontend uploads a file, `buildWorkspaceUploadPath` shall resolve the conversation's working dir through the same home-directory anchor, so the upload destination always matches the conversation's worktree location.
- [x] When the working dir is already absolute (e.g. POSIX `/foo`, Windows `C:\foo`, or the explicit selection from `search_subdirs`), the resolver shall pass it through unchanged.
- [x] The home-directory lookup shall be cached per backend host so concurrent uploads share a single in-flight `/api/file/home` request, and a cached value is reused for subsequent uploads.
- [x] A failed lookup shall not be cached so the next call retries fresh.
- [x] Upload paths shall never be constructed by naively prepending `/`; the legacy `toAbsoluteWorkspacePath` helper that did so was removed once its last callers disappeared (recoverable from git history), and the home-anchored resolver is the only sanctioned mechanism.

### Why this exists
- The agent-server's `/api/file/upload` endpoint requires an absolute path and `mkdir -p`s the parent of the destination. Naively prepending `/` to the default `workspace/project/<hex>` produces `/workspace/project/<hex>`. On macOS and on fresh Docker images that mount only `/home/<user>` as writable, the filesystem root is read-only, so the upload fails with `OSError: [Errno 30] Read-only file system: '/workspace'`.
- The agent-server otherwise interprets a relative `workspace.working_dir` against its process CWD (which is whichever directory the launcher used), so absent this resolver, the conversation's worktree lands in one place and the upload tries to land in a totally different place.
- `/api/file/home` is the most reliable absolute, writable anchor the agent-server API currently exposes; `/server_info` does not include the CWD.
