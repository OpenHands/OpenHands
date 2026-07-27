import { Provider } from "#/types/settings";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

const STORAGE_KEY = "openhands-agent-server-conversation-metadata";

export type WorkspaceMode = "local_repo" | "new_worktree";

export interface ConversationMetadata {
  selected_repository: string | null;
  selected_branch: string | null;
  git_provider: Provider | null;
  /**
   * The local workspace path the user explicitly attached at conversation
   * creation time. Distinct from `selected_repository` (which is set by
   * the repo picker on the home page). Used by the Files tab to decide
   * whether to default to diff view: if the user attached *anything*
   * (repo or local workspace), we lean diff-first because there's a real
   * git baseline to compare against.
   */
  selected_workspace?: string | null;
  /**
   * How the conversation should use `selected_workspace`.
   *
   * `local_repo` means the runtime should operate directly in the selected
   * folder, even when it is not a git checkout. `new_worktree` preserves the
   * historical agent-server behavior for conversations that should start in a
   * generated per-conversation worktree.
   */
  workspace_mode?: WorkspaceMode | null;
  /**
   * The LLM profile the conversation was created with (or last switched to).
   * Client-side only. Lets the chat-header switcher show the exact profile
   * name even when several profiles share the same underlying model — the
   * agent-server only round-trips the model string, so matching on it alone
   * is ambiguous (issue #1082).
   */
  active_profile?: string | null;
  /** Store plugin coordinates only; parameters may contain secrets. */
  plugins?: PluginSpec[] | null;
}

export const toPluginCoordinates = (plugin: PluginSpec): PluginSpec => ({
  source: plugin.source,
  ref: plugin.ref ?? null,
  repo_path: plugin.repo_path ?? null,
});

type StoredMetadata = Record<string, ConversationMetadata>;

const readAll = (): StoredMetadata => {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return {};
    return parsed as StoredMetadata;
  } catch {
    return {};
  }
};

const writeAll = (next: StoredMetadata): void => {
  if (typeof window === "undefined") return;
  if (Object.keys(next).length === 0) {
    window.localStorage.removeItem(STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
};

export const getStoredConversationMetadata = (
  conversationId: string,
): ConversationMetadata | null => readAll()[conversationId] ?? null;

export const setStoredConversationMetadata = (
  conversationId: string,
  metadata: ConversationMetadata,
): void => {
  const all = readAll();
  all[conversationId] = metadata;
  writeAll(all);
};

export const removeStoredConversationMetadata = (
  conversationId: string,
): void => {
  const all = readAll();
  if (!(conversationId in all)) return;
  delete all[conversationId];
  writeAll(all);
};
