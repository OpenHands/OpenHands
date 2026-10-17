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

---

### WUP-002: Upload destinations prefer the runtime-reported conversation working dir
- [x] When the frontend uploads a file, `uploadFilesToConversation` shall read the conversation's own `workspace.working_dir` from the runtime the upload targets (`GET /conversations/{id}` via `ConversationClient`) and, when that value is absolute, use it as the upload anchor.
- [x] The runtime value shall be adopted only when it is absolute; a relative runtime value shall be ignored, because the guard resolves it against the server's own process directory while Canvas resolves it against `/api/file/home`, so the two can disagree.
- [x] The locally-derived working dir (from `resolveConversationUploadWorkingDir`) shall remain the fallback used when the runtime read fails, returns an empty working dir, or returns a relative one. This preserves an already-valid absolute conversation workspace (e.g. an attached conversation's absolute `working_dir`) instead of replacing it with a relative value the guard would reject, so a transient runtime error never regresses the WUP-001 home-anchored behavior.
- [x] The runtime read shall be best-effort: any failure (network, 404, auth, sandbox still starting) falls back to the local resolution rather than failing the upload.
- [x] When an upload is rejected by the agent-server's workspace guard with `422` and the detail `path must be inside the conversation workspace`, the skipped-file reason shown to the user shall be a human-readable message (via `FILE_SERVICE$UPLOAD_WORKSPACE_MISMATCH`) instead of the raw `HTTP request failed (422 …)` body.

### Why this exists
- The agent-server's `bind_local_conversation_runtime` guard requires the upload `path` to be absolute and, after `.resolve()`, relative to `Path(conversation.workspace.working_dir).resolve()`. A *guessed* anchor (the relative `DEFAULT_WORKING_DIR` resolved against `/api/file/home`, or an assumed Cloud `/workspace/project`) can diverge from the dir the server actually stored, so the destination falls outside the workspace and the guard rejects it with `422 path must be inside the conversation workspace`.
- Reading the runtime's own value makes the destination authoritative instead of guessed, which is what the guard compares against; the local resolution is only a fallback for when the read is unavailable.
