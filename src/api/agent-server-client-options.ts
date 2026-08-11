import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
import { stripTrailingSlashes } from "./backend-registry/backend-host";
import type { Backend } from "./backend-registry/types";

export interface AgentServerClientOverrides {
  host?: string;
  apiKey?: string | null;
  sessionApiKey?: string | null;
  workingDir?: string;
  conversationUrl?: string | null;
  timeout?: number;
}

export interface AgentServerClientOptions {
  host: string;
  apiKey?: string;
  workingDir: string;
  timeout?: number;
}

export class NoBackendAvailableError extends Error {
  constructor() {
    super("No backend is configured.");
    this.name = "NoBackendAvailableError";
  }
}

export const isNoBackendAvailableError = (
  error: unknown,
): error is NoBackendAvailableError =>
  error instanceof NoBackendAvailableError ||
  (typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "NoBackendAvailableError");

/**
 * Only the trailing-slash primitive is applied here, not the full
 * `normalizeHostInput` used by the backend form. The hosts reaching this
 * function are already absolute — persisted backends were normalised when
 * they were saved, and `conversationUrl` arrives as an absolute URL — so
 * re-inferring a scheme would rewrite a caller-supplied override rather than
 * normalise it. Sharing the primitive keeps the two paths from drifting.
 */
function resolveHost(
  overrides: AgentServerClientOverrides,
  backend: Backend | null,
): string {
  if (overrides.host) return stripTrailingSlashes(overrides.host);
  if (overrides.conversationUrl)
    return stripTrailingSlashes(buildHttpBaseUrl(overrides.conversationUrl));
  return stripTrailingSlashes(backend?.host ?? "");
}

export function getAgentServerClientOptions(
  overrides: AgentServerClientOverrides = {},
): AgentServerClientOptions {
  const backend = getEffectiveLocalBackend();
  if (!backend && !overrides.host && !overrides.conversationUrl) {
    throw new NoBackendAvailableError();
  }

  const apiKey =
    overrides.sessionApiKey ?? overrides.apiKey ?? backend?.apiKey ?? undefined;

  return {
    host: resolveHost(overrides, backend),
    ...(apiKey ? { apiKey } : {}),
    workingDir: overrides.workingDir ?? getAgentServerWorkingDir(),
    ...(overrides.timeout !== undefined ? { timeout: overrides.timeout } : {}),
  };
}

export function getAgentServerHttpClientOptions(
  overrides?: AgentServerClientOverrides,
) {
  const { host, apiKey, timeout } = getAgentServerClientOptions(overrides);
  return {
    baseUrl: host,
    ...(apiKey ? { apiKey } : {}),
    timeout: timeout ?? 60000,
  };
}
