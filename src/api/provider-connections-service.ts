/**
 * ProviderConnectionsService — connect an LLM vendor once with one key and pick
 * from its model catalog, instead of pasting a key into every per-model LLM
 * profile. The key is stored as a named secret server-side; the server never
 * returns it, only an `api_key_set` boolean (so there is no key handling here).
 *
 * Routing:
 * - local agent-server: raw `fetch` against `/api/llm/connections`. There is no
 *   typed client in @openhands/typescript-client for these endpoints yet, so we
 *   follow the LLMSubscriptionService / LLMBalanceService precedent and use raw
 *   fetch with path constants (keeps this file out of the ad-hoc-HTTP guard).
 * - cloud app-server: Provider Connections are local-first in this phase. The
 *   deploy mirror is a follow-up (see issue #15492), so on cloud we throw a
 *   clear "not available" error rather than silently routing to the wrong API.
 *
 * Mirrors the snake_case ↔ camelCase normalization + defensive reads used by
 * LLMSubscriptionService so a slightly-off server response degrades gracefully.
 *
 * Refs OpenHands/OpenHands#15492, Linear OSS-5295.
 */
import { getAgentServerClientOptions } from "./agent-server-client-options";
import { getActiveBackend } from "./backend-registry/active-store";
import {
  PROVIDER_CONNECTION_PATH,
  PROVIDER_CONNECTIONS_PATH,
  PROVIDER_CONNECTION_VALIDATE_PATH,
} from "#/constants/provider-connections";

export interface ProviderConnection {
  id: string;
  provider: string;
  label: string | null;
  /** Models the catalog granted for this key (may be empty until validated). */
  models: string[];
  createdAt: number | null;
  lastValidatedAt: number | null;
  /** True when a key is stored; the key itself is never returned. */
  apiKeySet: boolean;
}

export interface CreateConnectionRequest {
  provider: string;
  key: string;
  label?: string;
}

export interface UpdateConnectionRequest {
  key?: string;
  label?: string;
  models?: string[];
}

export interface ValidateConnectionResponse {
  id: string;
  provider: string;
  ok: boolean;
  /** Models the key actually grants, per the provider's catalog. */
  models: string[];
  error: string | null;
  validatedAt: number | null;
}

class ProviderConnectionsNotOnCloudError extends Error {
  constructor() {
    super(
      "Provider Connections are not available on cloud in this release. Connect from a local agent-server backend.",
    );
    this.name = "ProviderConnectionsNotOnCloudError";
  }
}

export function isProviderConnectionsNotOnCloudError(
  error: unknown,
): error is ProviderConnectionsNotOnCloudError {
  return (
    error instanceof ProviderConnectionsNotOnCloudError ||
    (typeof error === "object" &&
      error !== null &&
      "name" in error &&
      (error as { name: unknown }).name ===
        "ProviderConnectionsNotOnCloudError")
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const readString = (
  value: Record<string, unknown>,
  keys: string[],
): string | null => {
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  return null;
};

const readNumber = (
  value: Record<string, unknown>,
  keys: string[],
): number | null => {
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      return candidate;
    }
  }
  return null;
};

const readBool = (value: Record<string, unknown>, keys: string[]): boolean => {
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "boolean") return candidate;
  }
  return false;
};

const readStringArray = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }
  if (isRecord(value) && Array.isArray(value.models)) {
    return value.models.filter(
      (item): item is string => typeof item === "string",
    );
  }
  return [];
};

function normalizeConnection(raw: unknown): ProviderConnection {
  if (!isRecord(raw)) {
    throw new Error("Provider connection response was not an object");
  }
  const id = readString(raw, ["id"]);
  const provider = readString(raw, ["provider"]);
  if (!id || !provider) {
    throw new Error("Provider connection response is missing id or provider");
  }
  return {
    id,
    provider,
    label: readString(raw, ["label"]),
    models: readStringArray(raw.models ?? raw["models_list"]),
    createdAt: readNumber(raw, ["created_at", "createdAt"]),
    lastValidatedAt: readNumber(raw, [
      "last_validated_at",
      "lastValidatedAt",
      "validated_at",
      "validatedAt",
    ]),
    apiKeySet: readBool(raw, ["api_key_set", "apiKeySet"]),
  };
}

function normalizeValidate(raw: unknown): ValidateConnectionResponse {
  if (!isRecord(raw)) {
    throw new Error("Validate connection response was not an object");
  }
  const id = readString(raw, ["id"]);
  const provider = readString(raw, ["provider"]);
  if (!id || !provider) {
    throw new Error("Validate connection response is missing id or provider");
  }
  return {
    id,
    provider,
    ok: readBool(raw, ["ok", "valid", "is_valid"]),
    models: readStringArray(raw),
    error: readString(raw, ["error", "message"]),
    validatedAt: readNumber(raw, ["validated_at", "validatedAt"]),
  };
}

function isCloudBackend(): boolean {
  return getActiveBackend().backend.kind === "cloud";
}

/** Throw early on cloud so callers can gate the UI before a network attempt. */
export function assertConnectionsSupportedLocally(): void {
  if (isCloudBackend()) throw new ProviderConnectionsNotOnCloudError();
}

async function requestConnectionEndpoint<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  assertConnectionsSupportedLocally();
  const { host, apiKey } = getAgentServerClientOptions();
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (apiKey) headers.set("X-Session-API-Key", apiKey);

  const response = await fetch(`${host}${path}`, { ...init, headers });
  if (!response.ok) {
    let detail = `Provider connection request failed with ${response.status}`;
    try {
      const body = (await response.json()) as Record<string, unknown>;
      const msg = readString(body, ["detail", "message", "error"]);
      if (msg) detail = msg;
    } catch {
      /* keep the status-based message */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

class ProviderConnectionsService {
  static async listConnections(): Promise<ProviderConnection[]> {
    const raw = await requestConnectionEndpoint<unknown[]>(
      PROVIDER_CONNECTIONS_PATH,
    );
    return (Array.isArray(raw) ? raw : []).map(normalizeConnection);
  }

  static async createConnection(
    request: CreateConnectionRequest,
  ): Promise<ProviderConnection> {
    const body: Record<string, unknown> = {
      provider: request.provider,
      key: request.key,
    };
    if (request.label) body.label = request.label;
    const raw = await requestConnectionEndpoint<unknown>(
      PROVIDER_CONNECTIONS_PATH,
      { method: "POST", body: JSON.stringify(body) },
    );
    return normalizeConnection(raw);
  }

  static async getConnection(id: string): Promise<ProviderConnection> {
    const raw = await requestConnectionEndpoint<unknown>(
      PROVIDER_CONNECTION_PATH(id),
    );
    return normalizeConnection(raw);
  }

  static async updateConnection(
    id: string,
    request: UpdateConnectionRequest,
  ): Promise<ProviderConnection> {
    const body: Record<string, unknown> = {};
    if (request.key !== undefined) body.key = request.key;
    if (request.label !== undefined) body.label = request.label;
    if (request.models !== undefined) body.models = request.models;
    const raw = await requestConnectionEndpoint<unknown>(
      PROVIDER_CONNECTION_PATH(id),
      { method: "PATCH", body: JSON.stringify(body) },
    );
    return normalizeConnection(raw);
  }

  static async deleteConnection(id: string): Promise<void> {
    await requestConnectionEndpoint<unknown>(PROVIDER_CONNECTION_PATH(id), {
      method: "DELETE",
    });
  }

  static async validateConnection(
    id: string,
  ): Promise<ValidateConnectionResponse> {
    const raw = await requestConnectionEndpoint<unknown>(
      PROVIDER_CONNECTION_VALIDATE_PATH(id),
      { method: "POST" },
    );
    return normalizeValidate(raw);
  }
}

export default ProviderConnectionsService;
