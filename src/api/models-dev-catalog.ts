import type { ACPModelOption } from "#/constants/acp-providers";

/**
 * Dynamic fallback model catalog sourced from https://models.dev/api.json.
 *
 * ACP providers (`src/constants/acp-providers.ts`) ship a small, curated
 * `available_models` list per provider. This service backs three things the
 * curated list alone can't:
 *
 *  1. Custom ACP servers (`ACP_CUSTOM_PRESET_KEY`) have no curated list at
 *     all — {@link getCatalogModelsForAcpServer} gives them something.
 *  2. Built-in providers get *more* models than the curated list covers —
 *     {@link mergeModelOptions} appends catalog-only entries.
 *  3. Effort/reasoning metadata (`reasoning_options`) the curated
 *     `ACPModelOption` shape doesn't carry — see {@link getEffortValuesForModel}.
 *
 * The upstream payload is ~3.5MB and covers every provider models.dev knows
 * about, most of which Canvas never renders. {@link fetchModelsDevCatalog}
 * trims it down to {@link ModelsDevModel} fields immediately after parsing
 * and only that trimmed shape is cached — the raw payload is never retained.
 *
 * This module is intentionally UI-free and does not throw: every public
 * function degrades to `null` / `[]` on bad input or network failure so a
 * models.dev outage can never break the ACP model picker, only shrink it
 * back to the curated list.
 */

const MODELS_DEV_API_URL = "https://models.dev/api.json";

/** localStorage key the cached catalog is stored under. Exported for tests. */
export const MODELS_DEV_CATALOG_STORAGE_KEY = "models-dev-catalog-v1";

/** How long a cached catalog is served without revalidating against models.dev. */
export const MODELS_DEV_CATALOG_TTL_MS = 1000 * 60 * 60 * 24;

/**
 * Maps an ACP registry key (`ACPProviderConfig.key`, e.g. `"claude-code"`)
 * to the models.dev provider id that carries its models
 * (`api.json[<id>].models`). Built-in providers only — a custom ACP server
 * has no fixed brand and must pass its models.dev provider key explicitly
 * (see {@link getCatalogModelsForAcpServer}'s `opts.providerKey`).
 */
export const MODELS_DEV_PROVIDER_BY_ACP_SERVER: Record<string, string> = {
  "claude-code": "anthropic",
  codex: "openai",
  "gemini-cli": "google",
};

/**
 * Resolve an ACP registry key to its models.dev provider id.
 *
 * Returns `null` for the `"custom"` preset and any key
 * {@link MODELS_DEV_PROVIDER_BY_ACP_SERVER} doesn't know about — callers
 * treat that as "no default catalog mapping" rather than an error.
 */
export function getModelsDevProviderKey(acpServer: string): string | null {
  return MODELS_DEV_PROVIDER_BY_ACP_SERVER[acpServer] ?? null;
}

/** Trimmed per-model fields kept from the models.dev payload. */
export interface ModelsDevModel {
  /** Exact model id, e.g. `"claude-opus-4-6"`. */
  id: string;
  /** Human-readable label — models.dev's `name` field. */
  label: string;
  description?: string;
  /**
   * Values of the `reasoning_options` entry whose `type` is `"effort"`
   * (e.g. `["low", "medium", "high"]`). Omitted when the model has no
   * effort-style reasoning control.
   */
  efforts?: string[];
  /** `limit.context` from the upstream payload, when present. */
  contextLimit?: number;
}

/** Trimmed per-provider slice of the models.dev catalog. */
export interface ModelsDevProviderCatalog {
  id: string;
  name: string;
  models: ModelsDevModel[];
}

/** The full trimmed catalog, keyed by models.dev provider id. */
export type ModelsDevCatalog = Record<string, ModelsDevProviderCatalog>;

function isValidModel(value: unknown): value is ModelsDevModel {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  if (typeof v.id !== "string" || typeof v.label !== "string") return false;
  if (v.description !== undefined && typeof v.description !== "string")
    return false;
  if (
    v.efforts !== undefined &&
    (!Array.isArray(v.efforts) ||
      !v.efforts.every((entry) => typeof entry === "string"))
  )
    return false;
  if (v.contextLimit !== undefined && typeof v.contextLimit !== "number")
    return false;
  return true;
}

function isValidProviderCatalog(
  value: unknown,
): value is ModelsDevProviderCatalog {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    typeof v.name === "string" &&
    Array.isArray(v.models) &&
    v.models.every(isValidModel)
  );
}

function isValidCatalog(value: unknown): value is ModelsDevCatalog {
  if (!value || typeof value !== "object") return false;
  return Object.values(value as Record<string, unknown>).every(
    isValidProviderCatalog,
  );
}

interface CachedCatalogEntry {
  /** `ETag` response header captured on the last successful 200, if any. */
  etag: string | null;
  /** `Date.now()` when this entry was last confirmed fresh (200 or 304). */
  fetchedAt: number;
  catalog: ModelsDevCatalog;
}

function isValidCacheEntry(value: unknown): value is CachedCatalogEntry {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    (v.etag === null || typeof v.etag === "string") &&
    typeof v.fetchedAt === "number" &&
    isValidCatalog(v.catalog)
  );
}

// In-memory fallback for the cache entry. Serves two purposes: (1) when
// `localStorage.setItem` throws (quota exceeded / storage disabled), the
// freshly-fetched catalog is still usable for the rest of the session; (2)
// it's the only cache available under SSR (`typeof window === "undefined"`).
// Reset between tests via `__resetModelsDevCatalogCacheForTests`.
let inMemoryCacheEntry: CachedCatalogEntry | null = null;

function readCacheEntry(): CachedCatalogEntry | null {
  if (typeof window !== "undefined") {
    try {
      const raw = window.localStorage.getItem(MODELS_DEV_CATALOG_STORAGE_KEY);
      if (raw) {
        // localStorage is user-editable — a hand-edited or stale-shape
        // entry is discarded rather than trusted.
        const parsed: unknown = JSON.parse(raw);
        if (isValidCacheEntry(parsed)) return parsed;
        return inMemoryCacheEntry;
      }
    } catch {
      // Corrupted JSON or storage access blocked — fall back below.
    }
  }
  return inMemoryCacheEntry;
}

function writeCacheEntry(entry: CachedCatalogEntry): void {
  inMemoryCacheEntry = entry;
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      MODELS_DEV_CATALOG_STORAGE_KEY,
      JSON.stringify(entry),
    );
  } catch {
    // Quota exceeded or storage disabled — already degraded to the
    // in-memory fallback above.
  }
}

/** Test-only: clear the in-memory cache fallback between test cases. */
export function __resetModelsDevCatalogCacheForTests(): void {
  inMemoryCacheEntry = null;
}

function extractEfforts(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const effortEntry = value.find(
    (entry) =>
      entry &&
      typeof entry === "object" &&
      (entry as Record<string, unknown>).type === "effort",
  ) as Record<string, unknown> | undefined;
  if (!effortEntry) return undefined;
  const { values } = effortEntry;
  if (!Array.isArray(values)) return undefined;
  const strings = values.filter(
    (entry): entry is string => typeof entry === "string",
  );
  return strings.length > 0 ? strings : undefined;
}

function extractContextLimit(limit: unknown): number | undefined {
  if (!limit || typeof limit !== "object") return undefined;
  const { context } = limit as Record<string, unknown>;
  return typeof context === "number" ? context : undefined;
}

function trimModel(fallbackId: string, value: unknown): ModelsDevModel | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  const id = typeof v.id === "string" && v.id ? v.id : fallbackId;
  const label = typeof v.name === "string" && v.name ? v.name : id;

  const model: ModelsDevModel = { id, label };
  if (typeof v.description === "string") model.description = v.description;

  const efforts = extractEfforts(v.reasoning_options);
  if (efforts) model.efforts = efforts;

  const contextLimit = extractContextLimit(v.limit);
  if (contextLimit !== undefined) model.contextLimit = contextLimit;

  return model;
}

function trimProvider(
  fallbackId: string,
  value: unknown,
): ModelsDevProviderCatalog | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  const id = typeof v.id === "string" && v.id ? v.id : fallbackId;
  const name = typeof v.name === "string" && v.name ? v.name : id;

  const models: ModelsDevModel[] = [];
  if (v.models && typeof v.models === "object") {
    for (const [modelKey, modelValue] of Object.entries(
      v.models as Record<string, unknown>,
    )) {
      const trimmed = trimModel(modelKey, modelValue);
      if (trimmed) models.push(trimmed);
    }
  }

  return { id, name, models };
}

/**
 * Trim the raw ~3.5MB models.dev payload down to {@link ModelsDevCatalog}.
 * Every provider is kept (a custom ACP server can map to any of them) but
 * only the compact per-model fields survive — the raw payload is discarded
 * right after this runs and never cached.
 *
 * Defensive against malformed entries: an individual bad provider/model is
 * skipped rather than failing the whole parse. Returns `null` only when the
 * top-level payload itself isn't a JSON object.
 */
export function trimModelsDevCatalog(raw: unknown): ModelsDevCatalog | null {
  if (!raw || typeof raw !== "object") return null;
  const catalog: ModelsDevCatalog = {};
  for (const [providerKey, providerValue] of Object.entries(
    raw as Record<string, unknown>,
  )) {
    const trimmed = trimProvider(providerKey, providerValue);
    if (trimmed) catalog[providerKey] = trimmed;
  }
  return catalog;
}

/**
 * Fetch the models.dev catalog, serving a 24h localStorage cache when
 * possible and revalidating with `If-None-Match` once it goes stale.
 *
 * Never throws — every failure path (network error, non-OK status, bad
 * JSON) resolves to the last known-good cache if one exists, or `null` if
 * not. Callers should treat `null` exactly like "no catalog available" and
 * fall back to whatever curated list they already have.
 */
export async function fetchModelsDevCatalog(): Promise<ModelsDevCatalog | null> {
  const cached = readCacheEntry();
  const now = Date.now();

  if (cached && now - cached.fetchedAt < MODELS_DEV_CATALOG_TTL_MS) {
    return cached.catalog;
  }

  try {
    const headers: HeadersInit = {};
    if (cached?.etag) {
      headers["If-None-Match"] = cached.etag;
    }

    const response = await fetch(MODELS_DEV_API_URL, { headers });

    if (response.status === 304 && cached) {
      // Server confirmed our cached copy is still current — refresh the
      // TTL clock without re-parsing anything.
      writeCacheEntry({ ...cached, fetchedAt: now });
      return cached.catalog;
    }

    if (!response.ok) {
      return cached?.catalog ?? null;
    }

    const raw: unknown = await response.json();
    const trimmed = trimModelsDevCatalog(raw);
    if (!trimmed) {
      return cached?.catalog ?? null;
    }

    writeCacheEntry({
      etag: response.headers.get("etag"),
      fetchedAt: now,
      catalog: trimmed,
    });
    return trimmed;
  } catch {
    // Network failure (offline, CORS, timeout, JSON parse error) — serve
    // whatever we had, even if stale, rather than blanking the picker.
    return cached?.catalog ?? null;
  }
}

/**
 * The trimmed models for the ACP provider `acpServer` maps to.
 *
 * `opts.providerKey` overrides the default {@link getModelsDevProviderKey}
 * lookup — the only way a custom ACP server (which has no entry in
 * {@link MODELS_DEV_PROVIDER_BY_ACP_SERVER}) can get a catalog list.
 *
 * Returns `[]` when there's no catalog, no resolvable provider key, or the
 * resolved key isn't present in `catalog`.
 */
export function getCatalogModelsForAcpServer(
  catalog: ModelsDevCatalog | null | undefined,
  acpServer: string,
  opts: { providerKey?: string } = {},
): ModelsDevModel[] {
  if (!catalog) return [];
  const providerKey = opts.providerKey ?? getModelsDevProviderKey(acpServer);
  if (!providerKey) return [];
  return catalog[providerKey]?.models ?? [];
}

export type ModelOptionSource = "curated" | "models.dev";

/**
 * An {@link ACPModelOption} tagged with where it came from, plus whatever
 * catalog metadata {@link mergeModelOptions} could attach.
 */
export interface MergedModelOption extends ACPModelOption {
  source: ModelOptionSource;
  description?: string;
  efforts?: string[];
}

/**
 * Merge the provider's curated model list with catalog models, curated
 * entries first (order preserved) followed by any catalog model whose id
 * isn't already curated.
 *
 * Dedupe is by exact id match after trimming — a catalog model with the
 * same id as a curated entry is dropped rather than duplicated, since the
 * curated entry is the maintained, human-reviewed one.
 */
export function mergeModelOptions(
  curated: ACPModelOption[],
  catalogModels: ModelsDevModel[],
): MergedModelOption[] {
  const curatedIds = new Set(curated.map((model) => model.id.trim()));

  const curatedMerged: MergedModelOption[] = curated.map(
    (model): MergedModelOption => ({
      ...model,
      source: "curated",
    }),
  );

  const catalogMerged: MergedModelOption[] = catalogModels
    .filter((model) => !curatedIds.has(model.id.trim()))
    .map(
      (model): MergedModelOption => ({
        id: model.id,
        label: model.label,
        source: "models.dev",
        description: model.description,
        efforts: model.efforts,
      }),
    );

  return [...curatedMerged, ...catalogMerged];
}

/**
 * The effort/reasoning values (e.g. `["low", "medium", "high"]`) models.dev
 * reports for `modelId` under `providerKey`, or `null` when the catalog,
 * provider, model, or effort metadata isn't available.
 */
export function getEffortValuesForModel(
  catalog: ModelsDevCatalog | null | undefined,
  providerKey: string,
  modelId: string,
): string[] | null {
  if (!catalog) return null;
  const model = catalog[providerKey]?.models.find((m) => m.id === modelId);
  return model?.efforts ?? null;
}
