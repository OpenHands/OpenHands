import { isOpenHandsCloudHost } from "#/api/device-flow-client";
import type { BackendKind } from "./types";

/**
 * Strip trailing slashes from a host string.
 *
 * This is the single trailing-slash policy for backend hosts. Both
 * {@link normalizeHostInput} (raw form input) and `getAgentServerClientOptions`
 * (already-persisted hosts) go through it, so the two paths cannot drift into
 * subtly different trimming rules.
 */
export function stripTrailingSlashes(host: string): string {
  return host.replace(/\/+$/, "");
}

/**
 * Returns true for hostnames that represent a local / private-network address.
 * Used by {@link normalizeHostInput} to choose http:// instead of https://.
 */
export function isLocalAddress(hostname: string): boolean {
  // Strip IPv6 bracket notation: [::1] → ::1
  const h = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  // IPv6 loopback, any-address, and named loopback
  if (h === "localhost" || h === "::1" || h === "::" || h === "0.0.0.0")
    return true;
  // 127.x.x.x loopback range + IPv4-mapped loopback (::ffff:127.x.x.x)
  if (/^127\./.test(h) || /^::ffff:127\./i.test(h)) return true;
  // RFC 1918 private ranges
  if (/^10\./.test(h)) return true;
  if (/^192\.168\./.test(h)) return true;
  if (/^172\.(1[6-9]|2\d|3[01])\./.test(h)) return true;
  // IPv6 link-local (fe80::/10) and unique local (fc00::/7)
  if (/^fe[89ab][0-9a-f]:/i.test(h)) return true;
  if (/^f[cd][0-9a-f]{2}:/i.test(h)) return true;
  // mDNS / Bonjour (.local)
  if (h.endsWith(".local")) return true;
  // Single-label hostnames (no dots, no colons) are local network names.
  // Colons are excluded so bare IPv6 addresses don't accidentally match.
  if (!h.includes(".") && !h.includes(":")) return true;
  return false;
}

/**
 * Normalise raw host input from the backend form into an absolute URL.
 *
 * This is deliberately richer than {@link stripTrailingSlashes}: it also trims
 * surrounding whitespace and prepends a scheme, because the form accepts bare
 * hosts such as `localhost:8000`. `getAgentServerClientOptions` intentionally
 * applies only the trailing-slash primitive — the hosts it sees have already
 * been through this function at save time, so re-inferring a scheme there
 * would rewrite a caller-supplied override rather than normalise it.
 */
export function normalizeHostInput(host: string): string {
  const trimmed = stripTrailingSlashes(host.trim());
  if (!trimmed) return "";
  // Already has an explicit scheme — respect it.
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  // Extract the pure hostname for scheme selection, handling three cases:
  //   [::1]:8080  → bracket IPv6 notation → extract ::1
  //   ::1         → bare IPv6 (multiple colons, no bracket) → whole string
  //   host:port   → regular host:port → part before the colon
  const bracketMatch = trimmed.match(/^\[([^\]]+)\]/);
  const hostname = bracketMatch
    ? bracketMatch[1]
    : (trimmed.match(/:/g) ?? []).length > 1
      ? trimmed
      : trimmed.split(":")[0];
  const scheme = isLocalAddress(hostname) ? "http" : "https";
  return `${scheme}://${trimmed}`;
}

/**
 * Returns true when `host` represents a reachable backend URL.
 *
 * Rules (applied in order):
 *   1. Must be non-empty after trimming.
 *   2. Must contain no whitespace — spaces can never appear in a host/port.
 *   3. After normalisation (bare hosts get a scheme prepended), must parse as
 *      a valid http or https URL with a non-empty hostname.
 */
export function isValidHostUrl(host: string): boolean {
  const trimmed = host.trim();
  if (!trimmed) return false;
  // Spaces anywhere in the input are an immediate rejection.
  if (/\s/.test(trimmed)) return false;
  const normalized = normalizeHostInput(trimmed);
  if (!normalized) return false;
  try {
    const url = new URL(normalized);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.hostname.length > 0
    );
  } catch {
    return false;
  }
}

/**
 * Seed the default backend kind from the host. Uses proper hostname-suffix
 * matching (via {@link isOpenHandsCloudHost}) rather than a substring test, so
 * a look-alike host such as `all-hands-testing.dev` isn't misread as cloud.
 *
 * This is only a *default*: a self-hosted OpenHands Cloud/Enterprise instance
 * on a truly custom domain is indistinguishable from a local agent-server by
 * host alone, so the manual add form lets the user override the kind
 * explicitly (see the Type selector in ManualConnectionColumn).
 */
export function inferKindFromHost(host: string): BackendKind {
  return isOpenHandsCloudHost(host) ? "cloud" : "local";
}
