// How long the CLI waits for one browser daemon call.
//
// Every verb answers within a default limit, but a verb's own deadline
// (`--timeout` on wait/text/attr…, `--observe-ms` on click) may run past it.
// The client then waits that long plus a margin, so a long wait ends with
// Playwright's TimeoutError and a failure screenshot, not an aborted request
// while the page action keeps running.

export const DEFAULT_CALL_LIMIT_MS = 120_000;
export const CALL_LIMIT_MARGIN_MS = 10_000;

export function browserCallLimit(args = {}, base = DEFAULT_CALL_LIMIT_MS) {
  const asked = Math.max(
    Number(args?.timeout) || 0,
    Number(args?.observeMs) || 0,
  );
  return Math.max(base, asked + CALL_LIMIT_MARGIN_MS);
}
