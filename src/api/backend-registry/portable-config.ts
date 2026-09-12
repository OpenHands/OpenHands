import type {
  Backend,
  BackendAuthMode,
  BackendInput,
  BackendKind,
} from "./types";

const PORTABLE_BACKEND_CONFIG_VERSION = 1;
export const PORTABLE_BACKEND_CONFIG_FILENAME = "openhands-backends.json";

interface PortableBackend {
  name: string;
  url: string;
  sessionApiKey: string;
  kind: BackendKind;
  authMode?: BackendAuthMode;
}

interface PortableBackendConfig {
  version: typeof PORTABLE_BACKEND_CONFIG_VERSION;
  backends: PortableBackend[];
}

function isBackendKind(value: unknown): value is BackendKind {
  return value === "local" || value === "cloud";
}

function isBackendAuthMode(value: unknown): value is BackendAuthMode {
  return value === "api-key" || value === "cookie";
}

function normalizeBackendUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, "");
  if (!trimmed || /\s/.test(trimmed)) {
    throw new Error("Invalid backend URL");
  }

  const parsed = new URL(trimmed);
  if (
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.hostname.length === 0
  ) {
    throw new Error("Invalid backend URL");
  }

  return trimmed;
}

function parsePortableBackend(value: unknown): PortableBackend {
  if (typeof value !== "object" || value === null) {
    throw new Error("Invalid backend entry");
  }

  const candidate = value as Partial<PortableBackend>;
  if (
    typeof candidate.name !== "string" ||
    candidate.name.trim().length === 0 ||
    typeof candidate.url !== "string" ||
    typeof candidate.sessionApiKey !== "string" ||
    !isBackendKind(candidate.kind) ||
    (candidate.authMode !== undefined && !isBackendAuthMode(candidate.authMode))
  ) {
    throw new Error("Invalid backend entry");
  }

  return {
    name: candidate.name.trim(),
    url: normalizeBackendUrl(candidate.url),
    sessionApiKey: candidate.sessionApiKey,
    kind: candidate.kind,
    ...(candidate.authMode === undefined
      ? {}
      : { authMode: candidate.authMode }),
  };
}

function toBackendInput(backend: PortableBackend): BackendInput {
  return {
    name: backend.name,
    host: backend.url,
    apiKey: backend.sessionApiKey,
    kind: backend.kind,
    ...(backend.authMode === undefined ? {} : { authMode: backend.authMode }),
  };
}

function canonicalBackendUrl(value: string): string {
  return new URL(normalizeBackendUrl(value)).toString();
}

export function serializePortableBackendConfig(backends: Backend[]): string {
  const config: PortableBackendConfig = {
    version: PORTABLE_BACKEND_CONFIG_VERSION,
    backends: backends.map((backend) => ({
      name: backend.name,
      url: backend.host,
      sessionApiKey: backend.apiKey,
      kind: backend.kind,
      ...(backend.authMode === undefined ? {} : { authMode: backend.authMode }),
    })),
  };

  return `${JSON.stringify(config, null, 2)}\n`;
}

export function parsePortableBackendConfig(raw: string): BackendInput[] {
  const parsed: unknown = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error("Invalid backend configuration");
  }

  const candidate = parsed as Partial<PortableBackendConfig>;
  if (
    candidate.version !== PORTABLE_BACKEND_CONFIG_VERSION ||
    !Array.isArray(candidate.backends)
  ) {
    throw new Error("Invalid backend configuration");
  }

  return candidate.backends.map(parsePortableBackend).map(toBackendInput);
}

export function mergePortableBackends(
  existing: Backend[],
  imported: BackendInput[],
  generateId: () => string,
): Backend[] {
  const merged = [...existing];

  for (const backend of imported) {
    const matchIndex = merged.findIndex(
      (candidate) =>
        canonicalBackendUrl(candidate.host) ===
        canonicalBackendUrl(backend.host),
    );

    if (matchIndex === -1) {
      merged.push({ ...backend, id: generateId() });
      continue;
    }

    const current = merged[matchIndex];
    const connectionChanged =
      current.host !== backend.host ||
      current.apiKey !== backend.apiKey ||
      current.kind !== backend.kind ||
      current.authMode !== backend.authMode;
    merged[matchIndex] = {
      ...backend,
      id: current.id,
      ...(connectionChanged
        ? { connectionRevision: (current.connectionRevision ?? 0) + 1 }
        : current.connectionRevision === undefined
          ? {}
          : { connectionRevision: current.connectionRevision }),
    };
  }

  return merged;
}
