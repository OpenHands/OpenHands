import { ServerClient } from "@openhands/typescript-client/clients";
import { getCachedAgentServerInfo } from "./agent-server-compatibility";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
import type { Backend } from "./backend-registry/types";

export const CONVERSATION_RUNTIME_CLIENT_UPGRADE_MESSAGE =
  "This Canvas build cannot access isolated Docker conversations safely. Upgrade Canvas to a build with a TypeScript client supporting conversation runtime routes.";

export function supportsConversationRuntimeRoutes(): boolean {
  return (
    "supportsConversationRuntimeRoutes" in ServerClient &&
    ServerClient.supportsConversationRuntimeRoutes === true
  );
}

export function assertConversationRuntimeClientSupport(): void {
  if (!supportsConversationRuntimeRoutes()) {
    throw new Error(CONVERSATION_RUNTIME_CLIENT_UPGRADE_MESSAGE);
  }
}

export interface AgentServerClientOverrides {
  conversationId?: string;
  host?: string;
  apiKey?: string | null;
  sessionApiKey?: string | null;
  workingDir?: string;
  conversationUrl?: string | null;
  timeout?: number;
}

export interface AgentServerClientOptions {
  conversationId?: string;
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

  const conversationId =
    overrides.conversationId ??
    overrides.conversationUrl?.match(
      /\/api\/conversations\/([^/?#]+)(?:[/?#]|$)/,
    )?.[1];

  const host = resolveHost(overrides, backend);
  const info = getCachedAgentServerInfo({ host });
  if (
    conversationId &&
    (info?.conversation_runtime === "docker" ||
      info?.workspace_mode === "isolated")
  ) {
    assertConversationRuntimeClientSupport();
  }

  return {
    host,
    ...(conversationId ? { conversationId } : {}),
    ...(apiKey ? { apiKey } : {}),
    workingDir: overrides.workingDir ?? getAgentServerWorkingDir(),
    ...(overrides.timeout !== undefined ? { timeout: overrides.timeout } : {}),
  };
}

export function getAgentServerHttpClientOptions(
  overrides?: AgentServerClientOverrides,
) {
  const { host, apiKey, timeout, conversationId } =
    getAgentServerClientOptions(overrides);
  return {
    baseUrl: host,
    ...(conversationId ? { conversationId } : {}),
    ...(apiKey ? { apiKey } : {}),
    timeout: timeout ?? 60000,
  };
}
