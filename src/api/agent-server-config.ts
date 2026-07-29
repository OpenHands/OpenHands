export const DEFAULT_WORKING_DIR = "workspace/project";

export type LockedCloudAuthMode = "api-key" | "cookie";

export interface AgentServerFormDefaults {
  baseUrl: string;
  sessionApiKey: string;
}

// Window-global key the static server injects `--lock-to-cloud` into; kept
// module-private because only `getLockedCloudHost()` reads it. The static
// server (`scripts/static-server.mjs`) and its tests reference the literal
// string directly, not this constant.
const LOCK_TO_CLOUD_WINDOW_KEY = "__AGENT_CANVAS_LOCK_TO_CLOUD__";
const LEGACY_CLOUD_DOMAIN = "all-hands.dev";
const CURRENT_CLOUD_DOMAIN = "openhands.dev";
const LEGACY_PRODUCTION_APP_HOST = `app.${LEGACY_CLOUD_DOMAIN}`;
const CURRENT_PRODUCTION_APP_HOST = CURRENT_CLOUD_DOMAIN;
const PRODUCTION_APP_HOST_ALIAS = `app.${CURRENT_CLOUD_DOMAIN}`;

function trimToNull(value?: string | null): string | null {
  return value?.trim() || null;
}

function normalizeBaseUrl(value?: string | null): string | null {
  if (!value) return null;

  const trimmed = value.trim().replace(/\/$/, "");
  if (!trimmed) return null;

  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed;
  }

  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${trimmed}`;
  }

  return `http://${trimmed}`;
}

function normalizeCloudHost(value?: string | null): string | null {
  if (!value) return null;

  const trimmed = value.trim().replace(/\/+$/, "");
  if (!trimmed) return null;

  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed;
  }

  return `https://${trimmed}`;
}

function canonicalizeCloudHostname(hostname: string): string {
  const lower = hostname.toLowerCase();
  if (
    lower === LEGACY_PRODUCTION_APP_HOST ||
    lower === PRODUCTION_APP_HOST_ALIAS
  ) {
    return CURRENT_PRODUCTION_APP_HOST;
  }

  if (lower === LEGACY_CLOUD_DOMAIN) return CURRENT_CLOUD_DOMAIN;
  if (lower.endsWith(`.${LEGACY_CLOUD_DOMAIN}`)) {
    return `${lower.slice(0, -LEGACY_CLOUD_DOMAIN.length)}${CURRENT_CLOUD_DOMAIN}`;
  }

  return lower;
}

function getCloudHostComparisonKey(value?: string | null): string | null {
  const normalized = normalizeCloudHost(value);
  if (!normalized) return null;

  try {
    const url = new URL(normalized);
    const port = url.port ? `:${url.port}` : "";
    return `${url.protocol}//${canonicalizeCloudHostname(url.hostname)}${port}`;
  } catch {
    return normalized.toLowerCase();
  }
}

export function getCookieAuthCloudHost(): string | null {
  const lockedHost = getLockedCloudHost();
  if (
    !lockedHost ||
    typeof window === "undefined" ||
    !isSameCloudHost(window.location.origin, lockedHost)
  ) {
    return null;
  }

  return window.location.origin;
}

function getConfiguredBaseUrl(): string | null {
  return normalizeBaseUrl(import.meta.env.VITE_BACKEND_BASE_URL);
}

/**
 * Return the session API key supplied by the deployment host.
 *
 * Two sources are consulted, in order:
 *   1. `VITE_SESSION_API_KEY` — baked into the bundle at build time (used by
 *      `npm run dev` so the dev server has the key without a round-trip).
 *   2. `window.__AGENT_CANVAS_SESSION_API_KEY__` — injected into `index.html`
 *      at serve time by `scripts/static-server.mjs --session-api-key <key>`.
 *      This is the path used by the published `agent-canvas` binary, where
 *      `VITE_SESSION_API_KEY` is empty in the prebuilt bundle and the
 *      runtime key is generated when the user launches the CLI.
 *
 * Without the window-global fallback, the published binary cannot construct a
 * default local backend (`makeDefaultLocalBackend()` returns null), the
 * registry is left empty, and the user sees the Manage Backends modal
 * instead of the onboarding flow.
 */
export function getBakedSessionApiKey(): string | null {
  const envKey = trimToNull(import.meta.env.VITE_SESSION_API_KEY);
  if (envKey) return envKey;

  if (typeof window !== "undefined") {
    const injected = (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SESSION_API_KEY__;
    if (typeof injected === "string") {
      return trimToNull(injected);
    }
  }

  return null;
}

export function getAgentServerFormDefaults(): AgentServerFormDefaults {
  return {
    baseUrl: getAgentServerBaseUrl() ?? "",
    sessionApiKey: getAgentServerSessionApiKey() ?? "",
  };
}

export function getLockedCloudHost(): string | null {
  const envHost = normalizeCloudHost(import.meta.env.VITE_LOCK_TO_CLOUD);
  if (envHost) return envHost;

  if (typeof window !== "undefined") {
    const injected = (window as unknown as Record<string, unknown>)[
      LOCK_TO_CLOUD_WINDOW_KEY
    ];
    if (typeof injected === "string") {
      return normalizeCloudHost(injected);
    }
  }

  return null;
}

/**
 * Compare a backend host against the locked Cloud host, normalizing
 * trailing slashes, protocol, and case so that e.g.
 * `https://app.all-hands.dev/` matches `https://app.all-hands.dev`.
 *
 * Used by the locked-to-Cloud gates (`root.tsx`,
 * `onboarding-modal.tsx`) to decide whether the active backend is the
 * configured locked Cloud host — a Cloud backend on a *different* host
 * (or a stale Local backend) must not be treated as the locked backend.
 */
export function isSameCloudHost(
  host: string | null | undefined,
  lockedHost: string | null | undefined,
): boolean {
  const a = getCloudHostComparisonKey(host);
  const b = getCloudHostComparisonKey(lockedHost);
  if (!a || !b) return false;
  return a === b;
}

export function getLockedCloudAuthMode(): LockedCloudAuthMode {
  return getCookieAuthCloudHost() ? "cookie" : "api-key";
}

export function getAgentServerBaseUrl(): string | null {
  const configuredUrl = getConfiguredBaseUrl();
  if (configuredUrl) return configuredUrl;

  if (typeof window !== "undefined") {
    return window.location.origin;
  }

  return null;
}

export function getAgentServerSessionApiKey(): string | null {
  return getBakedSessionApiKey();
}

export function getAgentServerWorkingDir(): string {
  const envDir = import.meta.env.VITE_WORKING_DIR?.trim();
  if (envDir) return envDir;

  return DEFAULT_WORKING_DIR;
}

export function buildConversationWorkingDir(conversationId: string): string {
  const base = getAgentServerWorkingDir().replace(/\/+$/, "");
  const hex = conversationId.replace(/-/g, "");
  return `${base}/${hex}`;
}

export function getAgentServerHeaders(): Record<string, string> {
  const sessionApiKey = getAgentServerSessionApiKey();
  return sessionApiKey ? { "X-Session-API-Key": sessionApiKey } : {};
}

export function isAuthRequired(): boolean {
  return (
    import.meta.env.VITE_AUTH_REQUIRED === "true" ||
    (typeof window !== "undefined" &&
      (window as unknown as Record<string, unknown>)
        .__AGENT_CANVAS_AUTH_REQUIRED__ === true)
  );
}

export function isAuthRequiredAndMissing(): boolean {
  if (!isAuthRequired()) return false;
  return !getAgentServerSessionApiKey();
}
