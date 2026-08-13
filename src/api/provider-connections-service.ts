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
  PROVIDER_CONNECTION_PROFILES_PATH,
  PROVIDER_CONNECTION_VALIDATE_PATH,
} from "#/constants/provider-connections";

export interface ProviderConnection {
  id: string;
  provider: string;
  label: string | null;
  baseUrl: string | null;
  apiMode: "auto" | "chat" | "responses";
  customHeaders: Record<string, string>;
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
  baseUrl?: string;
  apiMode?: "auto" | "chat" | "responses";
  customHeaders?: Record<string, string>;
  models?: string[];
}

export interface UpdateConnectionRequest {
  key?: string;
  label?: string;
  baseUrl?: string;
  apiMode?: "auto" | "chat" | "responses";
  customHeaders?: Record<string, string>;
  models?: string[];
}

export interface ValidateConnectionResponse {
  id: string;
  provider: string;
  ok: boolean;
  /**
   * True only when the key was checked against the provider over the network.
   * When false, `models` is the provider's advertised catalog rather than a
   * proven grant, so the UI must not claim the key was authenticated.
   */
  verified: boolean;
  /** Models the key actually grants, per the provider's catalog. */
  models: string[];
  error: string | null;
  validatedAt: number | null;
}

export interface DisconnectResponse {
  id: string;
  /** LLM profiles that referenced the deleted connection's key. */
  affectedProfiles: string[];
}

export interface CreateProfileFromConnectionRequest {
  profileName: string;
  model: string;
  baseUrl?: string;
}

export interface ProfileFromConnectionResponse {
  profileName: string;
  model: string;
  provider: string;
  connectionId: string;
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

const readStringRecord = (value: unknown): Record<string, string> => {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    ),
  );
};

const readApiMode = (value: unknown): "auto" | "chat" | "responses" => {
  if (value === "chat" || value === "responses") return value;
  return "auto";
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
    baseUrl: readString(raw, ["base_url", "baseUrl"]),
    apiMode: readApiMode(raw["api_mode"] ?? raw.apiMode),
    customHeaders: readStringRecord(raw["custom_headers"] ?? raw.customHeaders),
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
    verified: readBool(raw, ["verified"]),
    models: readStringArray(raw),
    error: readString(raw, ["error", "message"]),
    validatedAt: readNumber(raw, ["validated_at", "validatedAt"]),
  };
}

function normalizeDisconnect(
  raw: unknown,
  fallbackId: string,
): DisconnectResponse {
  if (!isRecord(raw)) {
    return { id: fallbackId, affectedProfiles: [] };
  }
  return {
    id: readString(raw, ["id"]) ?? fallbackId,
    affectedProfiles: readStringArray(
      raw.affected_profiles ?? raw.affectedProfiles,
    ),
  };
}

function normalizeProfileFromConnection(
  raw: unknown,
): ProfileFromConnectionResponse {
  if (!isRecord(raw)) {
    throw new Error(
      "Create-profile-from-connection response was not an object",
    );
  }
  const profileName = readString(raw, ["profile_name", "profileName"]);
  const model = readString(raw, ["model"]);
  const provider = readString(raw, ["provider"]);
  const connectionId = readString(raw, ["connection_id", "connectionId"]);
  if (!profileName || !model || !provider || !connectionId) {
    throw new Error(
      "Create-profile-from-connection response is missing required fields",
    );
  }
  return { profileName, model, provider, connectionId };
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
    if (request.baseUrl) body.base_url = request.baseUrl;
    if (request.apiMode) body.api_mode = request.apiMode;
    if (request.customHeaders) body.custom_headers = request.customHeaders;
    if (request.models) body.models = request.models;
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
    if (request.baseUrl !== undefined) body.base_url = request.baseUrl;
    if (request.apiMode !== undefined) body.api_mode = request.apiMode;
    if (request.customHeaders !== undefined) {
      body.custom_headers = request.customHeaders;
    }
    if (request.models !== undefined) body.models = request.models;
    const raw = await requestConnectionEndpoint<unknown>(
      PROVIDER_CONNECTION_PATH(id),
      { method: "PATCH", body: JSON.stringify(body) },
    );
    return normalizeConnection(raw);
  }

  static async deleteConnection(id: string): Promise<DisconnectResponse> {
    const raw = await requestConnectionEndpoint<unknown>(
      PROVIDER_CONNECTION_PATH(id),
      { method: "DELETE" },
    );
    return normalizeDisconnect(raw, id);
  }

  /**
   * Validate a connection's key. Pass `live` to force a real network probe;
   * the response's `verified` flag reflects whether that happened.
   */
  static async validateConnection(
    id: string,
    options?: { live?: boolean },
  ): Promise<ValidateConnectionResponse> {
    const path = options?.live
      ? `${PROVIDER_CONNECTION_VALIDATE_PATH(id)}?live=true`
      : PROVIDER_CONNECTION_VALIDATE_PATH(id);
    const raw = await requestConnectionEndpoint<unknown>(path, {
      method: "POST",
    });
    return normalizeValidate(raw);
  }

  /**
   * Create an LLM profile bound to a connection's key by reference (the
   * profile stores `secret:<name>` rather than the raw key, so rotating the
   * connection updates every profile at once). One call per selected model.
   */
  static async createProfileFromConnection(
    id: string,
    request: CreateProfileFromConnectionRequest,
  ): Promise<ProfileFromConnectionResponse> {
    const body: Record<string, unknown> = {
      profile_name: request.profileName,
      model: request.model,
    };
    if (request.baseUrl) body.base_url = request.baseUrl;
    const raw = await requestConnectionEndpoint<unknown>(
      PROVIDER_CONNECTION_PROFILES_PATH(id),
      { method: "POST", body: JSON.stringify(body) },
    );
    return normalizeProfileFromConnection(raw);
  }
}

export default ProviderConnectionsService;
