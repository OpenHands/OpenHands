import {
  ServerClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import type { ServerInfo as BaseServerInfo } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { isAuthRequired } from "#/api/agent-server-config";
import {
  getActiveBackend,
  getEffectiveLocalBackend,
  isNoBackend,
} from "#/api/backend-registry/active-store";
import defaults from "../../config/defaults.json";

const AGENT_SERVER_INFO_TIMEOUT_MS = 5000;
const UNKNOWN_AGENT_SERVER_VERSION = "unknown";

export const MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION =
  defaults.compatibility.minimumAgentServer;
export const AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE =
  "AGENT_SERVER_UNSUPPORTED_VERSION";
export const AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE =
  "AGENT_SERVER_UNKNOWN_VERSION";

export interface AgentServerInfo extends BaseServerInfo {
  sdk_version?: string;
  usable_tools?: string[] | null;
}

let cachedAgentServerInfo: AgentServerInfo | null = null;

const getAdvertisedTools = (serverInfo: AgentServerInfo | null) => {
  if (Array.isArray(serverInfo?.usable_tools)) {
    return serverInfo.usable_tools;
  }
  return null;
};

export class AgentServerUnavailableError extends Error {
  readonly details: string | null;
  readonly noBackendConfigured: boolean;

  constructor(
    details?: string | null,
    options?: { noBackendConfigured?: boolean },
  ) {
    const noBackendConfigured = options?.noBackendConfigured ?? false;
    super(
      noBackendConfigured
        ? "No agent server backend is configured yet. Add a backend to get started."
        : "Could not connect to the configured agent server. Make sure it is running and reachable, then reload the page.",
    );
    this.name = "AgentServerUnavailableError";
    this.details = details ?? null;
    this.noBackendConfigured = noBackendConfigured;
  }
}

export const isAgentServerUnavailableError = (
  error: unknown,
): error is AgentServerUnavailableError =>
  error instanceof AgentServerUnavailableError ||
  (typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "AgentServerUnavailableError");

export class AgentServerUnsupportedVersionError extends AgentServerUnavailableError {
  readonly code = AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE;
  readonly actualVersion: string;
  readonly requiredVersion = MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION;

  constructor(actualVersion: string) {
    const message = `Agent Canvas requires agent-server ${MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION} or newer; this backend is running ${actualVersion}. Please upgrade the agent-server backend.`;
    super(message);
    this.name = "AgentServerUnsupportedVersionError";
    this.message = message;
    this.actualVersion = actualVersion;
  }
}

export class AgentServerUnknownVersionError extends AgentServerUnavailableError {
  readonly code = AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE;
  readonly actualVersion: string | null;
  readonly requiredVersion = MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION;

  constructor(actualVersion: string | null) {
    const reported = actualVersion ? ` It reported "${actualVersion}".` : "";
    const message =
      `Could not determine this backend's agent-server version.${reported} ` +
      `Agent Canvas requires agent-server ${MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION} ` +
      "or newer, but this backend did not return a valid version from " +
      "/server_info. Restart or rebuild the agent-server backend, then try again.";
    super(message);
    this.name = "AgentServerUnknownVersionError";
    this.message = message;
    this.actualVersion = actualVersion;
  }
}

export const isAgentServerUnsupportedVersionError = (
  error: unknown,
): error is AgentServerUnsupportedVersionError =>
  error instanceof AgentServerUnsupportedVersionError ||
  (typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE);

export const isAgentServerUnknownVersionError = (
  error: unknown,
): error is AgentServerUnknownVersionError =>
  error instanceof AgentServerUnknownVersionError ||
  (typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE);

/**
 * Returns true when the agent-server probe failed with HTTP 401.
 * In public mode this means the stored key is stale (server restarted
 * with a different `LOCAL_BACKEND_API_KEY`). Only meaningful when
 * auth is required — a 401 in local mode is a misconfiguration, not a
 * key-rotation event. Uses {@link isAuthRequired} so both the build-time
 * `VITE_AUTH_REQUIRED` flag and the runtime `window.__AGENT_CANVAS_AUTH_REQUIRED__`
 * injection (used by pre-built static binaries) are honoured.
 */
export const isAgentServerAuthError = (error: unknown): boolean =>
  isAuthRequired() && isSdkHttpStatusError(error, 401);

export function clearCachedAgentServerInfo() {
  cachedAgentServerInfo = null;
}

export function isAgentServerToolAvailable(toolName: string) {
  const availableTools = getAdvertisedTools(cachedAgentServerInfo);
  if (!Array.isArray(availableTools)) {
    return true;
  }
  return availableTools.includes(toolName);
}

export function isSdkHttpError(error: unknown) {
  return (
    error instanceof Error &&
    error.name === "HttpError" &&
    "status" in error &&
    typeof error.status === "number"
  );
}

/**
 * Narrows an SDK HTTP error to a specific status code.
 * Use instead of manually casting `(err as { status: number }).status`.
 */
export function isSdkHttpStatusError(error: unknown, status: number): boolean {
  return (
    isSdkHttpError(error) && (error as { status: number }).status === status
  );
}

function normalizeAgentServerInfoVersion(version: unknown): string | null {
  if (typeof version !== "string") return null;
  const trimmed = version.trim();
  return trimmed || null;
}

function getRawAgentServerSdkVersion(
  serverInfo: AgentServerInfo,
): string | null {
  return normalizeAgentServerInfoVersion(serverInfo.sdk_version);
}

function getRawAgentServerVersion(serverInfo: AgentServerInfo): string | null {
  return (
    normalizeAgentServerInfoVersion(serverInfo.version) ??
    getRawAgentServerSdkVersion(serverInfo)
  );
}

function getComparableAgentServerVersion(
  serverInfo: AgentServerInfo,
): string | null {
  const version = getRawAgentServerVersion(serverInfo);
  if (!version || version.toLowerCase() === UNKNOWN_AGENT_SERVER_VERSION) {
    return null;
  }
  return version;
}

export function getDisplayAgentServerVersion(
  serverInfo: AgentServerInfo,
): string | null {
  const version = getComparableAgentServerVersion(serverInfo);
  if (!version || !parseAgentServerVersion(version)) {
    return null;
  }
  return version;
}

export function getDisplayAgentServerSdkVersion(
  serverInfo: AgentServerInfo,
): string | null {
  const version =
    getRawAgentServerSdkVersion(serverInfo) ??
    getComparableAgentServerVersion(serverInfo);
  if (!version || version.toLowerCase() === UNKNOWN_AGENT_SERVER_VERSION) {
    return null;
  }
  return version;
}

export function getCachedAgentServerSdkVersion(
  host?: string | null,
): string | null {
  if (!cachedAgentServerInfo) return null;
  if (host && getEffectiveLocalBackend()?.host !== host) return null;
  return getDisplayAgentServerSdkVersion(cachedAgentServerInfo);
}

export function getCachedAgentServerVersion(
  host?: string | null,
): string | null {
  if (!cachedAgentServerInfo) return null;
  if (host && getEffectiveLocalBackend()?.host !== host) return null;
  return getDisplayAgentServerVersion(cachedAgentServerInfo);
}

export function compareAgentServerVersions(actual: string, required: string) {
  const parsedActual = parseAgentServerVersion(actual);
  const parsedRequired = parseAgentServerVersion(required);

  if (!parsedActual || !parsedRequired) {
    return null;
  }

  for (const key of ["major", "minor", "patch"] as const) {
    if (parsedActual[key] > parsedRequired[key]) {
      return 1;
    }
    if (parsedActual[key] < parsedRequired[key]) {
      return -1;
    }
  }

  if (parsedActual.prerelease && !parsedRequired.prerelease) {
    return -1;
  }
  if (!parsedActual.prerelease && parsedRequired.prerelease) {
    return 1;
  }
  if (parsedActual.prerelease && parsedRequired.prerelease) {
    return parsedActual.prerelease.localeCompare(parsedRequired.prerelease);
  }

  return 0;
}

function parseAgentServerVersion(version: string) {
  const trimmed = version.trim().replace(/^v/, "");
  const [withoutBuild] = trimmed.split("+");
  const [core, prerelease] = withoutBuild.split("-", 2);
  const parts = core.split(".");

  if (parts.length !== 3) {
    return null;
  }

  const [major, minor, patch] = parts.map((part) => Number(part));
  if (
    ![major, minor, patch].every((part) => Number.isInteger(part) && part >= 0)
  ) {
    return null;
  }

  return { major, minor, patch, prerelease };
}

export function assertAgentServerVersionIsSupported(
  serverInfo: AgentServerInfo,
) {
  const actualVersion = getComparableAgentServerVersion(serverInfo);
  if (!actualVersion) {
    clearCachedAgentServerInfo();
    throw new AgentServerUnknownVersionError(
      getRawAgentServerVersion(serverInfo),
    );
  }

  const comparison = compareAgentServerVersions(
    actualVersion,
    MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
  );

  if (comparison === null) {
    clearCachedAgentServerInfo();
    throw new AgentServerUnknownVersionError(actualVersion);
  }

  if (comparison < 0) {
    clearCachedAgentServerInfo();
    throw new AgentServerUnsupportedVersionError(actualVersion);
  }
}

export async function loadAgentServerInfo() {
  // The probe is a *local* agent-server concern — it verifies the runtime
  // hosting the GUI is reachable. It must NEVER run against the active
  // backend when that backend is cloud, because cloud hosts don't
  // expose /api/server_info and would fail with a CORS error besides.
  const local = getEffectiveLocalBackend();
  if (!local) {
    clearCachedAgentServerInfo();

    // Empty registry (NO_BACKEND sentinel) — the user has no backend
    // configured at all.  Throw so root.tsx shows the manage-backends
    // modal instead of silently rendering a broken home page.
    if (isNoBackend(getActiveBackend().backend)) {
      throw new AgentServerUnavailableError("No backend configured", {
        noBackendConfigured: true,
      });
    }

    // Active backend is cloud — no local probe needed.
    return null;
  }

  const clientOptions = getAgentServerClientOptions({
    host: local.host,
    sessionApiKey: local.apiKey || null,
    timeout: AGENT_SERVER_INFO_TIMEOUT_MS,
  });
  let serverInfo: AgentServerInfo;

  try {
    serverInfo = (await new ServerClient(
      clientOptions,
    ).getServerInfo()) as AgentServerInfo;
  } catch (error) {
    clearCachedAgentServerInfo();
    // Preserve 401 so root.tsx can show the auth screen (public mode).
    // All other HTTP errors (502, 503, etc.) mean the server is unreachable
    // or misconfigured — treat them as unavailable.
    if (isSdkHttpStatusError(error, 401)) {
      throw error;
    }

    const details = error instanceof Error ? error.message : null;
    throw new AgentServerUnavailableError(details);
  }

  assertAgentServerVersionIsSupported(serverInfo);

  // /server_info is unprotected, so a stale session key still gets 200.
  // In public mode, validate the key against a protected endpoint so a
  // server restart with a new LOCAL_BACKEND_API_KEY surfaces immediately
  // instead of letting the app load and fail on every subsequent call.
  if (isAuthRequired()) {
    try {
      await new SettingsClient(clientOptions).getSettings();
    } catch (error) {
      // Only rethrow 401 — that means the stored key is invalid /
      // rotated.  Other HTTP errors (403, 5xx) and non-HTTP errors
      // (network, timeout) are swallowed: the server *is* up (we just
      // reached /server_info), so let the app proceed with an
      // unvalidated key rather than blocking the UI.
      // NOTE: If the connection drops between the /server_info and
      // getSettings() probes, the app loads with an unvalidated key and
      // subsequent 401s won't trigger the auth screen (they come from
      // React Query hooks, not this bootstrap path). Acceptable for now
      // since the window is narrow and a page refresh recovers.
      if (isSdkHttpStatusError(error, 401)) {
        throw error;
      }

      console.warn(
        "[agent-server] getSettings() probe failed (non-401):",
        error,
      );
    }
  }

  cachedAgentServerInfo = serverInfo;
  return serverInfo;
}
