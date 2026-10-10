/**
 * Narrow allowance for intentional in-process (loopback) traffic.
 *
 * The Vitest setup is fail-closed: `vitest.setup.ts` starts MSW with
 * `onUnhandledRequest: "error"`, so a request without a matching handler fails the
 * test that made it instead of escaping to the real network.
 *
 * Some suites are built around real servers on this machine.
 * `__tests__/scripts/ingress.test.ts` and `__tests__/scripts/static-server.test.ts`
 * listen on a loopback port and then fetch it. That traffic never leaves the
 * machine, so it keeps an explicit allowance here instead of a blanket remote bypass.
 *
 * The allowance is a passthrough handler rather than a custom `onUnhandledRequest`
 * callback for a concrete reason: under MSW's network frame registry the printer
 * handed to a callback only prints the report and does not resolve the frame
 * (`node_modules/msw/src/core/experimental/on-unhandled-frame.ts` says so), so a
 * callback could allow traffic but could not fail a single request — it would print
 * "unhandled" and let the request through anyway. A handler matches what is
 * intentional and leaves `"error"` in charge of everything else.
 *
 * Anything that is not loopback must be mocked with its own handler.
 */

import { http, passthrough } from "msw";
import type { HttpHandler } from "msw";

/**
 * Hostnames that identify intentional, in-process traffic.
 *
 * `URL` keeps IPv6 hosts bracketed, hence the bracketed spelling of the loopback
 * address. `0.0.0.0` is included because a request to it resolves to this machine.
 */
const LOOPBACK_HOSTNAMES = new Set([
  "localhost",
  "127.0.0.1",
  "0.0.0.0",
  "[::1]",
]);

/**
 * True when `url` targets this machine.
 *
 * Intent is decided from the parsed hostname, never from a substring match, so a
 * remote host that merely mentions a loopback address
 * (`https://evil.test/localhost`) is still treated as remote.
 */
export function isLoopbackUrl(url: string): boolean {
  let hostname: string;
  try {
    ({ hostname } = new URL(url));
  } catch {
    // An unparsable URL is not provably local, so it must not be allowed.
    return false;
  }
  return LOOPBACK_HOSTNAMES.has(hostname.toLowerCase());
}

/**
 * Passthrough handlers for loopback traffic.
 *
 * Spread last in `handlers.ts` so that an explicit mock for a loopback URL still
 * wins over this catch-all.
 */
export const LOOPBACK_PASSTHROUGH_HANDLERS: HttpHandler[] = [
  http.all(
    ({ request }) => isLoopbackUrl(request.url),
    () => passthrough(),
  ),
];
