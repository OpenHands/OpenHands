/**
 * Clerk configuration for the team-access gate.
 *
 * Agent Canvas ships with no user identity of its own — the only
 * credentials it holds are *machine* credentials for the agent server
 * (`X-Session-API-Key` / cloud bearer token, see
 * `#/api/backend-registry/auth`). Clerk sits in front of all of that and
 * answers a single question: is the person in this browser on the team?
 *
 * Everything here is opt-in. With no publishable key configured
 * `isClerkEnabled()` is false and the gate renders its children straight
 * through, which keeps the `npx @openhands/agent-canvas` local workflow,
 * the e2e suites, and OSS consumers on their existing behavior.
 *
 * > [!IMPORTANT]
 * > This gate protects the *canvas UI*, not the agent server. The agent
 * > server still authenticates on the session API key alone, so anyone
 * > holding that key can talk to it directly without ever meeting Clerk.
 * > See docs/TEAM_ACCESS.md.
 */

// Window globals mirror the `__AGENT_CANVAS_SESSION_API_KEY__` /
// `__AGENT_CANVAS_LOCK_TO_CLOUD__` pattern in `agent-server-config.ts`, so a
// static host can inject Clerk settings into a prebuilt bundle without a
// rebuild.
const PUBLISHABLE_KEY_WINDOW_KEY = "__AGENT_CANVAS_CLERK_PUBLISHABLE_KEY__";
const ALLOWED_DOMAINS_WINDOW_KEY = "__AGENT_CANVAS_CLERK_ALLOWED_DOMAINS__";

function trimToNull(value?: string | null): string | null {
  return value?.trim() || null;
}

function readWindowString(key: string): string | null {
  if (typeof window === "undefined") return null;
  const injected = (window as unknown as Record<string, unknown>)[key];
  return typeof injected === "string" ? trimToNull(injected) : null;
}

/**
 * The Clerk publishable key (`pk_test_…` / `pk_live_…`). Safe to ship in
 * the client bundle by design — it identifies the Clerk instance, it is
 * not a secret.
 */
export function getClerkPublishableKey(): string | null {
  return (
    trimToNull(import.meta.env.VITE_CLERK_PUBLISHABLE_KEY) ??
    readWindowString(PUBLISHABLE_KEY_WINDOW_KEY)
  );
}

export function isClerkEnabled(): boolean {
  return getClerkPublishableKey() !== null;
}

/**
 * Optional second layer on top of Clerk Dashboard restrictions: a
 * comma-separated email-domain allowlist (`acme.com, acme.dev`).
 *
 * The Dashboard is the real control — set sign-ups to *Restricted* and
 * invite the team, so a stranger can never mint a session in the first
 * place. This list only catches the case where that configuration drifts,
 * and it runs in the browser, so treat it as a safety net rather than a
 * boundary. Leave it unset to rely on the Dashboard alone.
 */
export function getClerkAllowedEmailDomains(): string[] {
  const raw =
    trimToNull(import.meta.env.VITE_CLERK_ALLOWED_EMAIL_DOMAINS) ??
    readWindowString(ALLOWED_DOMAINS_WINDOW_KEY);
  if (!raw) return [];

  return raw
    .split(",")
    .map((domain) => domain.trim().toLowerCase().replace(/^@/, ""))
    .filter(Boolean);
}

/**
 * Whether `email` is on the allowlist. An empty allowlist permits every
 * address — the Dashboard decides who exists at all.
 */
export function isEmailDomainAllowed(
  email: string | null | undefined,
  allowedDomains: string[] = getClerkAllowedEmailDomains(),
): boolean {
  if (allowedDomains.length === 0) return true;

  const domain = trimToNull(email)?.toLowerCase().split("@").pop();
  if (!domain) return false;

  return allowedDomains.includes(domain);
}
