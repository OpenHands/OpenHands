import type {
  Backend,
  BackendAuthMode,
  BackendKind,
} from "#/api/backend-registry/types";

export interface PortableBackend {
  name: string;
  host: string;
  apiKey: string;
  kind: BackendKind;
  authMode?: BackendAuthMode;
}

interface PortableBackendConfig {
  version: 1;
  backends: PortableBackend[];
}

function normalizeHost(host: string): string {
  const parsed = new URL(host.trim());
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Backend URLs must use http or https.");
  }
  parsed.hash = "";
  parsed.search = "";
  parsed.pathname = parsed.pathname.replace(/\/+$/, "") || "/";
  return parsed.toString().replace(/\/$/, "").toLowerCase();
}

function isBackendKind(value: unknown): value is BackendKind {
  return value === "local" || value === "cloud";
}

function isAuthMode(value: unknown): value is BackendAuthMode | undefined {
  return value === undefined || value === "api-key" || value === "cookie";
}

function parsePortableBackend(value: unknown): PortableBackend {
  if (typeof value !== "object" || value === null) {
    throw new Error("Each backend entry must be an object.");
  }
  const backend = value as Partial<PortableBackend>;
  if (
    typeof backend.name !== "string" ||
    backend.name.trim().length === 0 ||
    typeof backend.host !== "string" ||
    typeof backend.apiKey !== "string" ||
    !isBackendKind(backend.kind) ||
    !isAuthMode(backend.authMode)
  ) {
    throw new Error("The backend export file contains an invalid entry.");
  }

  normalizeHost(backend.host);
  return {
    name: backend.name.trim(),
    host: backend.host.trim().replace(/\/+$/, ""),
    apiKey: backend.apiKey,
    kind: backend.kind,
    ...(backend.authMode ? { authMode: backend.authMode } : {}),
  };
}

export function serializePortableBackends(backends: Backend[]): string {
  const payload: PortableBackendConfig = {
    version: 1,
    backends: backends.map(
      ({ name, host, apiKey, kind, authMode }): PortableBackend => ({
        name,
        host,
        apiKey,
        kind,
        ...(authMode ? { authMode } : {}),
      }),
    ),
  };
  return JSON.stringify(payload, null, 2);
}

export function parsePortableBackends(raw: string): PortableBackend[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("This is not a valid backend configuration JSON file.");
  }

  if (
    typeof parsed !== "object" ||
    parsed === null ||
    (parsed as Partial<PortableBackendConfig>).version !== 1 ||
    !Array.isArray((parsed as Partial<PortableBackendConfig>).backends)
  ) {
    throw new Error("This file is not an OpenHands backend export.");
  }

  return (parsed as PortableBackendConfig).backends.map(parsePortableBackend);
}

function defaultIdFactory(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `imported-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Merge imported bootstrap configuration by normalized backend URL.
 * Matching entries keep their stable id so active selections remain valid.
 */
export function mergePortableBackends(
  existing: Backend[],
  imported: PortableBackend[],
  idFactory: () => string = defaultIdFactory,
): Backend[] {
  const result = [...existing];

  for (const incoming of imported) {
    const incomingHost = normalizeHost(incoming.host);
    const index = result.findIndex(
      (backend) => normalizeHost(backend.host) === incomingHost,
    );

    if (index >= 0) {
      const current = result[index];
      const credentialsChanged =
        current.apiKey !== incoming.apiKey ||
        current.authMode !== incoming.authMode ||
        current.kind !== incoming.kind;
      result[index] = {
        ...current,
        ...incoming,
        connectionRevision: credentialsChanged
          ? (current.connectionRevision ?? 0) + 1
          : current.connectionRevision,
      };
      continue;
    }

    result.push({
      id: idFactory(),
      ...incoming,
    });
  }

  return result;
}
