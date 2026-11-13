import type { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * A sandbox whose runtime is gone and cannot be resumed. Set for a Cloud
 * `sandbox_status: "MISSING"` and for a local runtime the adapter maps from
 * `runtime_info.runtime_status: "missing"` with `can_resume: false`.
 *
 * `ERROR` is deliberately excluded: an errored runtime keeps its red "Error"
 * indicator and is not treated as archived.
 */
export function isMissingSandboxStatus(
  sandboxStatus: SandboxStatus | null | undefined,
): boolean {
  return sandboxStatus === "MISSING";
}

/**
 * Whether the conversation's runtime is unavailable, so runtime-backed
 * affordances (WebSocket, chat input, git/panel toggles) are read-only and
 * its persisted history is the only thing left to read. Covers both a missing
 * (non-resumable) sandbox and an errored one.
 *
 * This is a runtime concern, not an archive concern: do not use it to decide
 * list visibility, the "Archived" chip, or the archive/unarchive direction.
 */
export function isRuntimeUnavailableSandboxStatus(
  sandboxStatus: SandboxStatus | null | undefined,
): boolean {
  return sandboxStatus === "MISSING" || sandboxStatus === "ERROR";
}

/**
 * The single "effective archived" predicate: a conversation is archived when
 * the user explicitly archived it, or when its runtime is missing and
 * non-resumable. Drives the archived presentation (list visibility, the
 * "Archived" chip, the dimmed title, and the status dot) so those never
 * disagree.
 *
 * This is presentation only. Menu availability is a separate concern: a
 * missing, non-resumable runtime cannot be restored from the menu, so such a
 * row offers neither "Archive" nor "Unarchive" even when this returns true.
 */
export function isEffectivelyArchivedConversation(
  sandboxStatus: SandboxStatus | null | undefined,
  explicitlyArchived: boolean,
): boolean {
  return explicitlyArchived || isMissingSandboxStatus(sandboxStatus);
}
