/**
 * Remember which client tool calls have already been acted on.
 *
 * Unlike the Canvas UI tool, some client tools are not idempotent: replaying
 * their ActionEvent (a socket reconnect that falls back to `resend_mode:
 * "all"`, or a REST/WebSocket race after a reload) would repeat real work.
 * Claim the tool call before doing any so a replay that arrives mid-flight is
 * dropped too. Returns false when the call was already claimed.
 */
export function claimClientToolCall(ledgerKey: string, toolCallId: string) {
  let handled: string[] = [];
  try {
    const raw = window.localStorage.getItem(ledgerKey);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    if (Array.isArray(parsed)) {
      handled = parsed.filter((id): id is string => typeof id === "string");
    }
  } catch {
    // A corrupt ledger must not block the call; start a fresh one.
  }

  if (handled.includes(toolCallId)) return false;

  try {
    window.localStorage.setItem(
      ledgerKey,
      JSON.stringify([...handled, toolCallId]),
    );
  } catch {
    // Storage full or unavailable — proceed, accepting replay risk over never
    // acting at all.
  }
  return true;
}
