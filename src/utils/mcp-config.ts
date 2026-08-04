import type {
  MCPAuthCredential,
  MCPConfig,
  MCPConfigPatch,
  MCPServer,
  MCPServerPatch,
} from "@openhands/typescript-client";
import { isMcpAuthCredential } from "#/types/mcp-auth";
import type { MCPServerConfig } from "#/types/mcp-server";
import { toMcpServerName } from "#/utils/mcp-server-name";

export const REDACTED_MCP_SECRET_VALUE = "**********";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);

export function getSdkMcpServerMap(
  value: unknown,
): Record<string, unknown> | null {
  if (!isRecord(value)) return null;

  const mcpServers = value.mcpServers;
  if (
    isRecord(mcpServers) &&
    !("url" in mcpServers) &&
    !("command" in mcpServers)
  ) {
    return mcpServers;
  }

  return value;
}

export function stringRecord(
  value: unknown,
): Record<string, string> | undefined {
  if (!isRecord(value)) return undefined;
  const entries = Object.entries(value).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );
  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}

export function hasRedactedMcpSecretLeaf(value: unknown): boolean {
  if (value === REDACTED_MCP_SECRET_VALUE) return true;
  if (Array.isArray(value)) return value.some(hasRedactedMcpSecretLeaf);
  if (isRecord(value)) {
    return Object.values(value).some(hasRedactedMcpSecretLeaf);
  }
  return false;
}

// Agent Server persists this optional field, but typescript-client 1.36.1
// does not yet expose it in its generated MCP model types.
export const getMcpServerEnabled = (server: MCPServer): boolean | undefined =>
  (server as MCPServer & { enabled?: unknown }).enabled === false
    ? false
    : undefined;

const normalizeTransport = (
  value: unknown,
): MCPServer["transport"] | undefined => {
  if (value === "stdio" || value === "sse" || value === "streamable-http") {
    return value;
  }
  if (value === "http" || value === "shttp" || value === undefined) {
    return "http";
  }
  return undefined;
};

/**
 * Normalize the SDK and legacy cloud wrapper shapes into the client-owned,
 * name-keyed MCPConfig. The map key remains the server's persistence identity.
 */
// @spec MCP-003 — Settings map keys are stable MCP identities
export function parseMcpConfig(value: unknown): MCPConfig {
  const serverMap = getSdkMcpServerMap(value);
  if (!serverMap) return {};

  const config: MCPConfig = {};
  for (const [settingsKey, candidate] of Object.entries(serverMap)) {
    if (!isRecord(candidate)) continue;

    if (typeof candidate.command === "string") {
      config[settingsKey] = {
        transport: "stdio",
        command: candidate.command,
        ...(Array.isArray(candidate.args) &&
          candidate.args.every((arg) => typeof arg === "string") && {
            args: candidate.args as string[],
          }),
        ...(stringRecord(candidate.env) && {
          env: stringRecord(candidate.env),
        }),
        ...(typeof candidate.cwd === "string" && { cwd: candidate.cwd }),
        ...(typeof candidate.description === "string" && {
          description: candidate.description,
        }),
        ...(typeof candidate.icon === "string" && { icon: candidate.icon }),
        ...(typeof candidate.timeout === "number" && {
          timeout: candidate.timeout,
        }),
        ...(candidate.enabled === false && { enabled: false }),
      };
      continue;
    }

    if (typeof candidate.url !== "string") continue;
    const transport = normalizeTransport(candidate.transport);
    if (!transport || transport === "stdio") continue;
    config[settingsKey] = {
      transport,
      url: candidate.url,
      ...(stringRecord(candidate.headers) && {
        headers: stringRecord(candidate.headers),
      }),
      ...(isMcpAuthCredential(candidate.auth) && {
        auth: candidate.auth,
      }),
      ...(typeof candidate.description === "string" && {
        description: candidate.description,
      }),
      ...(typeof candidate.icon === "string" && { icon: candidate.icon }),
      ...(typeof candidate.timeout === "number" && {
        timeout: candidate.timeout,
      }),
      ...(typeof candidate.sse_read_timeout === "number" && {
        sse_read_timeout: candidate.sse_read_timeout,
      }),
      ...(typeof candidate.keep_alive === "boolean" && {
        keep_alive: candidate.keep_alive,
      }),
      ...(candidate.enabled === false && { enabled: false }),
    };
  }
  return config;
}

export function toCanonicalMcpServer(server: MCPServerConfig): MCPServer {
  if (server.type === "stdio") {
    return {
      transport: "stdio",
      command: server.command!,
      ...(server.args?.length ? { args: server.args } : {}),
      ...(server.env && Object.keys(server.env).length > 0
        ? { env: server.env }
        : {}),
      ...(server.enabled === false && { enabled: false }),
    };
  }

  return {
    transport: server.type === "sse" ? "sse" : "http",
    url: server.url!,
    ...(server.headers && Object.keys(server.headers).length > 0
      ? { headers: server.headers }
      : {}),
    ...(server.auth ? { auth: server.auth } : {}),
    ...(server.type === "shttp" && server.timeout !== undefined
      ? { timeout: server.timeout }
      : {}),
    ...(server.enabled === false && { enabled: false }),
  };
}

const buildEnabledPatch = (
  previous: MCPServer,
  edited: MCPServerConfig,
): MCPServerPatch => {
  if (edited.enabled === false) return { enabled: false } as MCPServerPatch;
  return getMcpServerEnabled(previous) === false
    ? ({ enabled: true } as MCPServerPatch)
    : {};
};

const buildStringMapPatch = (
  previous: Record<string, string> | null | undefined,
  next: Record<string, string> | undefined,
): Record<string, string | null> | null | undefined => {
  if (!next) return previous ? null : undefined;

  const patch: Record<string, string | null> = {};
  for (const [key, value] of Object.entries(next)) {
    if (value !== REDACTED_MCP_SECRET_VALUE) {
      patch[key] = value;
    }
  }
  for (const key of Object.keys(previous ?? {})) {
    if (!(key in next)) patch[key] = null;
  }
  return Object.keys(patch).length > 0 ? patch : undefined;
};

type OAuthCredential = Extract<MCPAuthCredential, { strategy: "oauth2" }>;
type OAuthCredentialPatch = Extract<
  NonNullable<MCPServerPatch["auth"]>,
  { strategy: "oauth2" }
>;
const OAUTH_EDITABLE_AUTHENTICATION_FIELDS = [
  "client_auth_method",
  "scopes",
  "client_id",
  "client_secret",
] as const;

const buildRedactionSafeNestedPatch = (
  previous: unknown,
  next: unknown,
): unknown | undefined => {
  if (next === REDACTED_MCP_SECRET_VALUE || next === undefined) {
    return undefined;
  }
  if (next === null) return null;
  if (Array.isArray(next)) {
    if (hasRedactedMcpSecretLeaf(next)) return undefined;
    return JSON.stringify(previous) === JSON.stringify(next) ? undefined : next;
  }
  if (!isRecord(next)) {
    return Object.is(previous, next) ? undefined : next;
  }

  const previousRecord = isRecord(previous) ? previous : {};
  const patch: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(next)) {
    const nestedPatch = buildRedactionSafeNestedPatch(
      previousRecord[key],
      value,
    );
    if (nestedPatch !== undefined) patch[key] = nestedPatch;
  }
  return Object.keys(patch).length > 0 ? patch : undefined;
};

const buildOAuthAuthenticationPatch = (
  previous: OAuthCredential["authentication"],
  edited: OAuthCredential["authentication"],
): OAuthCredentialPatch["authentication"] | undefined => {
  if (edited === undefined) return undefined;
  if (edited === null) return null;

  const nestedPatch = buildRedactionSafeNestedPatch(previous, edited);
  const patch = isRecord(nestedPatch) ? { ...nestedPatch } : {};
  const previousRecord = isRecord(previous) ? previous : {};
  for (const key of OAUTH_EDITABLE_AUTHENTICATION_FIELDS) {
    if (key in previousRecord && !(key in edited)) patch[key] = null;
  }
  if (Object.keys(patch).length === 0) return undefined;
  return { type: "oauth", ...patch } as NonNullable<
    OAuthCredentialPatch["authentication"]
  >;
};

const withAuthStrategyReplacementDeletes = <T extends MCPAuthCredential>(
  previous: MCPAuthCredential | null | undefined,
  patch: T,
): T => {
  if (!previous || previous.strategy === patch.strategy) return patch;

  const replacement = { ...patch } as Record<string, unknown>;
  for (const key of Object.keys(previous)) {
    if (key !== "strategy" && !(key in replacement)) {
      replacement[key] = null;
    }
  }
  return replacement as T;
};

type HeaderAuthCredential = Extract<MCPAuthCredential, { strategy: "header" }>;

export const MCP_HEADER_REMOVAL_ERROR =
  "Removing an individual header from header authentication is not supported yet. Replace the credential or clear authentication, then re-enter the headers you want to keep.";

const buildHeaderAuthCredentialPatch = (
  previous: MCPAuthCredential | null | undefined,
  edited: HeaderAuthCredential,
): NonNullable<MCPServerPatch["auth"]> | undefined => {
  const previousHeaders =
    previous?.strategy === "header" ? previous.headers : undefined;
  const headersPatch = buildStringMapPatch(previousHeaders, edited.headers);
  if (!headersPatch) return undefined;
  if (Object.values(headersPatch).some((value) => value === null)) {
    throw new Error(MCP_HEADER_REMOVAL_ERROR);
  }
  return {
    strategy: "header",
    headers: headersPatch as Record<string, string>,
  };
};

const buildOAuthCredentialPatch = (
  previous: MCPAuthCredential | null | undefined,
  edited: OAuthCredential,
): OAuthCredentialPatch | undefined => {
  const previousOAuth = previous?.strategy === "oauth2" ? previous : undefined;
  const authentication = buildOAuthAuthenticationPatch(
    previousOAuth?.authentication,
    edited.authentication,
  );
  const state = buildRedactionSafeNestedPatch(
    previousOAuth?.state,
    edited.state,
  );
  if (previousOAuth && authentication === undefined && state === undefined) {
    return undefined;
  }
  return withAuthStrategyReplacementDeletes(previous, {
    strategy: "oauth2",
    ...(authentication !== undefined && { authentication }),
    ...(state !== undefined && { state }),
  } as OAuthCredentialPatch);
};

/**
 * Build one sparse MCP server merge-patch from an editor result. Redacted
 * values are display-only and are never mutation inputs.
 */
// @spec MCP-002 — Secret patches preserve user intent
export function buildMcpServerPatch(
  previous: MCPServer,
  edited: MCPServerConfig,
): MCPServerPatch {
  const enabled = buildEnabledPatch(previous, edited);
  if (edited.type === "stdio") {
    const env = buildStringMapPatch(
      previous.transport === "stdio" ? previous.env : undefined,
      edited.env,
    );
    return {
      transport: "stdio",
      command: edited.command!,
      ...(edited.args?.length
        ? { args: edited.args }
        : previous.transport === "stdio" && previous.args
          ? { args: null }
          : {}),
      ...(env !== undefined ? { env } : {}),
      ...enabled,
    };
  }

  const previousRemote = previous.transport === "stdio" ? undefined : previous;
  const patch: MCPServerPatch = {
    transport: edited.type === "sse" ? "sse" : "http",
    url: edited.url!,
    ...enabled,
  };

  if (edited.type === "shttp") {
    if (edited.timeout !== undefined) patch.timeout = edited.timeout;
    else if (previousRemote?.timeout !== undefined) patch.timeout = null;
  }

  if (edited.auth) {
    if (edited.auth.strategy === "oauth2") {
      const auth = buildOAuthCredentialPatch(previousRemote?.auth, edited.auth);
      if (auth !== undefined) patch.auth = auth;
    } else if (
      edited.auth.strategy === "header" &&
      previousRemote?.auth?.strategy === "header"
    ) {
      const auth = buildHeaderAuthCredentialPatch(
        previousRemote.auth,
        edited.auth,
      );
      if (auth !== undefined) patch.auth = auth;
    } else if (!hasRedactedMcpSecretLeaf(edited.auth)) {
      patch.auth = withAuthStrategyReplacementDeletes(
        previousRemote?.auth,
        edited.auth,
      );
    }
  } else if (previousRemote?.auth) {
    patch.auth = null;
  }

  return patch;
}

const applyMcpServerPatch = (
  previous: MCPServer,
  patch: MCPServerPatch,
): MCPServer => {
  const apply = (base: unknown, next: unknown): Record<string, unknown> => {
    const merged = isRecord(base) ? { ...base } : {};
    if (!isRecord(next)) return merged;

    for (const [key, value] of Object.entries(next)) {
      if (value === null) {
        delete merged[key];
      } else if (isRecord(value)) {
        merged[key] = apply(merged[key], value);
      } else {
        merged[key] = value;
      }
    }
    return merged;
  };

  return apply(previous, patch) as MCPServer;
};

export const MCP_RENAME_CREDENTIAL_ERROR =
  "Replace or clear the stored credential before renaming this MCP server.";

// @spec MCP-003 — Settings map keys are stable MCP identities
export function buildRenameMcpConfigPatch(
  oldKey: string,
  newKey: string,
  previous: MCPServer,
  edited: MCPServerConfig,
): MCPConfigPatch {
  const previousSecrets =
    previous.transport === "stdio"
      ? previous.env
      : { auth: previous.auth, headers: previous.headers };
  if (hasRedactedMcpSecretLeaf(previousSecrets)) {
    throw new Error(MCP_RENAME_CREDENTIAL_ERROR);
  }
  const renamedServer = applyMcpServerPatch(
    previous,
    buildMcpServerPatch(previous, edited),
  );
  return {
    [oldKey]: null,
    [newKey]: renamedServer,
  };
}

export function allocateMcpSettingsKey(
  config: MCPConfig,
  preferredName: string | undefined,
  fallback: "sse" | "shttp" | "stdio",
): string {
  const base = toMcpServerName(preferredName || fallback);
  if (!(base in config)) return base;
  let suffix = 1;
  while (`${base}_${suffix}` in config) suffix += 1;
  return `${base}_${suffix}`;
}
