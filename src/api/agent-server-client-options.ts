import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
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

function normalizeHost(host: string): string {
  return host.replace(/\/+$/, "");
}

function resolveHost(
  overrides: AgentServerClientOverrides,
  backend: Backend | null,
): string {
  if (overrides.host) return normalizeHost(overrides.host);
  if (overrides.conversationUrl)
    return normalizeHost(buildHttpBaseUrl(overrides.conversationUrl));
  return normalizeHost(backend?.host ?? "");
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
