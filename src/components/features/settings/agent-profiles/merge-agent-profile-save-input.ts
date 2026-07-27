import type {
  AgentProfile,
  AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";

/**
 * Merge the minimal editor's fields over the stored profile so an edit-save
 * doesn't wipe the fields the editor doesn't model (condenser, verification,
 * system_message_suffix, mcp_server_refs, the disabled_skills deny-list, ACP
 * session mode/timeout, …) — `POST /api/agent-profiles/{name}` is a
 * whole-profile overwrite, and unset fields fall back to server-side defaults.
 *
 * Kind-aware: when the editor switched `agent_kind`, the stored variant's
 * fields must NOT be carried over — the server's `extra="forbid"` profile
 * union rejects a payload mixing openhands and acp fields — so a kind switch
 * stays a clean variant replacement built from the edited fields alone.
 *
 * Identity (`id` / `name` / `revision`) is stripped from the merge: the path
 * name is authoritative on save, and the server preserves the namesake's id
 * and bumps the revision itself (`save_profile_preserving_identity`).
 */
export function mergeAgentProfileSaveInput(
  stored: AgentProfile | null,
  edited: AgentProfileSaveInput,
): AgentProfileSaveInput {
  if (!stored) return edited;
  if (stored.agent_kind !== edited.agent_kind) return edited;
  const { id, name, revision, ...preserved } = stored;
  return { ...preserved, ...edited } as AgentProfileSaveInput;
}
