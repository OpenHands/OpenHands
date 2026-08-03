interface SlashCommandOutputScope {
  backendId: string;
  orgId: string | null | undefined;
  conversationId?: string | null;
}

/**
 * Slash-command cards are ephemeral UI state, but conversation IDs are not a
 * sufficient namespace: two configured backends (or two Cloud organizations)
 * can legitimately expose the same ID. Keep every producer and renderer on a
 * single collision-safe scope derived from the complete active selection.
 */
export function buildSlashCommandOutputScopeId({
  backendId,
  orgId,
  conversationId = null,
}: SlashCommandOutputScope): string {
  return JSON.stringify([
    "slash-command-output",
    backendId,
    orgId ?? null,
    conversationId,
  ]);
}
