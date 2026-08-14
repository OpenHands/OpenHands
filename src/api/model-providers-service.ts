/**
 * ModelProvidersService — connect an LLM provider once with one API key, then
 * manage the models under it (add / edit / remove). The key is held on the
 * provider and shared by every nested model, instead of pasting it into each
 * per-model LLM profile. The key is stored as a named secret server-side; the
 * server never returns it, only an `apiKeySet` boolean (so there is no key
 * handling here).
 *
 * Routing:
 * - local agent-server: raw `fetch` against `/api/llm/model-providers`. There
 *   is no typed client in @openhands/typescript-client for these endpoints yet,
 *   so we follow the LLMSubscriptionService / LLMBalanceService precedent and
 *   use raw fetch with path constants (keeps this file out of the ad-hoc-HTTP
 *   guard).
 * - cloud app-server: model providers are local-first in this phase. The
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
  MODEL_PROVIDER_MODEL_PATH,
  MODEL_PROVIDER_MODELS_PATH,
  MODEL_PROVIDER_PATH,
  MODEL_PROVIDER_TEST_PATH,
  MODEL_PROVIDERS_PATH,
} from "#/constants/model-providers";

export type WireApi = "auto" | "chat" | "responses";

export interface ProviderModel {
  name: string;
  /** Optional per-model wire-API override; falls back to the provider's. */
  wireApi: WireApi | null;
}

export interface ModelProvider {
  id: string;
  /** Vendor kind, e.g. "openai", "anthropic", "custom". */
  kind: string;
  displayName: string;
  baseUrl: string | null;
  wireApi: WireApi;
  customHeaders: Record<string, string>;
  /** Models the user curated under this provider (nested, editable). */
  models: ProviderModel[];
  createdAt: number | null;
  updatedAt: number | null;
  /** True when a key is stored; the key itself is never returned. */
  apiKeySet: boolean;
}

export interface CreateProviderRequest {
  kind: string;
  displayName: string;
  key: string;
  baseUrl?: string;
  wireApi?: WireApi;
  customHeaders?: Record<string, string>;
  models?: { name: string; wireApi?: WireApi | null }[];
}

export interface UpdateProviderRequest {
  displayName?: string;
  kind?: string;
  key?: string;
  baseUrl?: string;
  wireApi?: WireApi;
  customHeaders?: Record<string, string>;
}

export interface ModelPayload {
  name: string;
  wireApi?: WireApi | null;
}

export interface TestProviderResponse {
  id: string;
  ok: boolean;
  /**
   * True only when the key was checked against the provider over the network.
   * When false, `suggestedModels` is the provider's advertised catalog rather
   * than a proven grant, so the UI must not claim the key was authenticated.
   */
  verified: boolean;
  /** Catalog models offered as a convenience for the "add model" affordance. */
  suggestedModels: string[];
  error: string | null;
}

class ModelProvidersNotOnCloudError extends Error {
  constructor() {
    super(
      "Model providers are not available on cloud in this release. Connect from a local agent-server backend.",
    );
    this.name = "ModelProvidersNotOnCloudError";
  }
}

export function isModelProvidersNotOnCloudError(
  error: unknown,
): error is ModelProvidersNotOnCloudError {
  return (
    error instanceof ModelProvidersNotOnCloudError ||
    (typeof error === "object" &&
      error !== null &&
      "name" in error &&
      (error as { name: unknown }).name === "ModelProvidersNotOnCloudError")
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

const readWireApi = (value: unknown): WireApi => {
  if (value === "chat" || value === "responses") return value;
  return "auto";
};

const readModels = (value: unknown): ProviderModel[] => {
  if (!Array.isArray(value)) return [];
  const out: ProviderModel[] = [];
  for (const item of value) {
    if (typeof item === "string") {
      out.push({ name: item, wireApi: null });
    } else if (isRecord(item)) {
      const name = readString(item, ["name", "model"]);
      if (!name) continue;
      const rawWire = item["wire_api"] ?? item.wireApi;
      out.push({
        name,
        wireApi:
          rawWire === "chat" || rawWire === "responses" || rawWire === "auto"
            ? (rawWire as WireApi)
            : null,
      });
    }
  }
  return out;
};

function normalizeProvider(raw: unknown): ModelProvider {
  if (!isRecord(raw)) {
    throw new Error("Model provider response was not an object");
  }
  const id = readString(raw, ["id"]);
  const kind = readString(raw, ["kind", "provider"]);
  const displayName =
    readString(raw, ["display_name", "displayName", "label"]) ?? kind;
  if (!id || !kind || !displayName) {
    throw new Error("Model provider response is missing id, kind, or name");
  }
  return {
    id,
    kind,
    displayName,
    baseUrl: readString(raw, ["base_url", "baseUrl"]),
    wireApi: readWireApi(raw["wire_api"] ?? raw.wireApi),
    customHeaders: readStringRecord(raw["custom_headers"] ?? raw.customHeaders),
    models: readModels(raw.models),
    createdAt: readNumber(raw, ["created_at", "createdAt"]),
    updatedAt: readNumber(raw, ["updated_at", "updatedAt"]),
    apiKeySet: readBool(raw, ["api_key_set", "apiKeySet"]),
  };
}

function normalizeTest(raw: unknown): TestProviderResponse {
  if (!isRecord(raw)) {
    throw new Error("Test provider response was not an object");
  }
  const id = readString(raw, ["id"]);
  if (!id) throw new Error("Test provider response is missing id");
  return {
    id,
    ok: readBool(raw, ["ok", "valid", "is_valid"]),
    verified: readBool(raw, ["verified"]),
    suggestedModels: readStringArray(
      raw["suggested_models"] ?? raw.suggestedModels,
    ),
    error: readString(raw, ["error", "message"]),
  };
}

function isCloudBackend(): boolean {
  return getActiveBackend().backend.kind === "cloud";
}

/** Throw early on cloud so callers can gate the UI before a network attempt. */
export function assertProvidersSupportedLocally(): void {
  if (isCloudBackend()) throw new ModelProvidersNotOnCloudError();
}

async function requestProviderEndpoint<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  assertProvidersSupportedLocally();
  const { host, apiKey } = getAgentServerClientOptions();
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (apiKey) headers.set("X-Session-API-Key", apiKey);

  const response = await fetch(`${host}${path}`, { ...init, headers });
  if (!response.ok) {
    let detail = `Model provider request failed with ${response.status}`;
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

class ModelProvidersService {
  static async listProviders(): Promise<ModelProvider[]> {
    const raw = await requestProviderEndpoint<unknown[]>(MODEL_PROVIDERS_PATH);
    return (Array.isArray(raw) ? raw : []).map(normalizeProvider);
  }

  static async createProvider(
    request: CreateProviderRequest,
  ): Promise<ModelProvider> {
    const body: Record<string, unknown> = {
      kind: request.kind,
      display_name: request.displayName,
      key: request.key,
    };
    if (request.baseUrl) body.base_url = request.baseUrl;
    if (request.wireApi) body.wire_api = request.wireApi;
    if (request.customHeaders) body.custom_headers = request.customHeaders;
    if (request.models) {
      body.models = request.models.map((m) => ({
        name: m.name,
        wire_api: m.wireApi ?? null,
      }));
    }
    const raw = await requestProviderEndpoint<unknown>(MODEL_PROVIDERS_PATH, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return normalizeProvider(raw);
  }

  static async getProvider(id: string): Promise<ModelProvider> {
    const raw = await requestProviderEndpoint<unknown>(MODEL_PROVIDER_PATH(id));
    return normalizeProvider(raw);
  }

  static async updateProvider(
    id: string,
    request: UpdateProviderRequest,
  ): Promise<ModelProvider> {
    const body: Record<string, unknown> = {};
    if (request.displayName !== undefined) {
      body.display_name = request.displayName;
    }
    if (request.kind !== undefined) body.kind = request.kind;
    if (request.key !== undefined) body.key = request.key;
    if (request.baseUrl !== undefined) body.base_url = request.baseUrl;
    if (request.wireApi !== undefined) body.wire_api = request.wireApi;
    if (request.customHeaders !== undefined) {
      body.custom_headers = request.customHeaders;
    }
    const raw = await requestProviderEndpoint<unknown>(
      MODEL_PROVIDER_PATH(id),
      {
        method: "PATCH",
        body: JSON.stringify(body),
      },
    );
    return normalizeProvider(raw);
  }

  static async deleteProvider(id: string): Promise<ModelProvider> {
    const raw = await requestProviderEndpoint<unknown>(
      MODEL_PROVIDER_PATH(id),
      {
        method: "DELETE",
      },
    );
    return normalizeProvider(raw);
  }

  static async addModel(
    id: string,
    model: ModelPayload,
  ): Promise<ModelProvider> {
    const raw = await requestProviderEndpoint<unknown>(
      MODEL_PROVIDER_MODELS_PATH(id),
      {
        method: "POST",
        body: JSON.stringify({
          name: model.name,
          wire_api: model.wireApi ?? null,
        }),
      },
    );
    return normalizeProvider(raw);
  }

  static async updateModel(
    id: string,
    modelName: string,
    model: ModelPayload,
  ): Promise<ModelProvider> {
    const raw = await requestProviderEndpoint<unknown>(
      MODEL_PROVIDER_MODEL_PATH(id, modelName),
      {
        method: "PATCH",
        body: JSON.stringify({
          name: model.name,
          wire_api: model.wireApi ?? null,
        }),
      },
    );
    return normalizeProvider(raw);
  }

  static async removeModel(
    id: string,
    modelName: string,
  ): Promise<ModelProvider> {
    const raw = await requestProviderEndpoint<unknown>(
      MODEL_PROVIDER_MODEL_PATH(id, modelName),
      { method: "DELETE" },
    );
    return normalizeProvider(raw);
  }

  /**
   * Probe the provider's stored key. `verified` reflects whether a real
   * network check happened; `suggestedModels` is a catalog convenience for the
   * add-model affordance and never mutates the curated model list.
   */
  static async testProvider(id: string): Promise<TestProviderResponse> {
    const raw = await requestProviderEndpoint<unknown>(
      MODEL_PROVIDER_TEST_PATH(id),
      { method: "POST" },
    );
    return normalizeTest(raw);
  }
}

export default ModelProvidersService;
