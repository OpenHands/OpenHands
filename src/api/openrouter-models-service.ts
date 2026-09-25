export const OPENROUTER_PROVIDER = "openrouter";
const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
const CATALOG_TIMEOUT_MS = 5000;

/**
 * Display-only capability metadata for an OpenRouter catalog model. Every
 * field is best-effort: OpenRouter may omit any of them, and an absent field
 * means "unknown" — never "unsupported". Callers must not use this to block
 * or filter a user's model choice, only to show useful context.
 */
export interface OpenRouterModelCapabilities {
  /** Raw `supported_parameters` values, e.g. `["tools", "reasoning"]`. */
  supportedParameters?: string[];
  inputModalities?: string[];
  outputModalities?: string[];
  contextLength?: number;
  /** Provider-reported max completion/output tokens, when published. */
  maxOutputTokens?: number;
  /** Per-token price strings as returned by OpenRouter, display-only. */
  pricing?: {
    prompt?: string;
    completion?: string;
  };
}

export interface OpenRouterCatalogModel extends OpenRouterModelCapabilities {
  id: string;
}

/**
 * An explicit empty array is a *known* value (e.g. OpenRouter published
 * `supported_parameters: []`, meaning "no parameters" rather than "didn't
 * say") and must be preserved as `[]`, not folded into "unknown". A missing
 * or non-array value, or one whose entries are all non-strings, is genuinely
 * unknown and stays `undefined`.
 */
function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  if (value.length === 0) return [];
  const strings = value.filter(
    (item): item is string => typeof item === "string",
  );
  return strings.length > 0 ? strings : undefined;
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function readPricing(value: unknown): OpenRouterModelCapabilities["pricing"] {
  if (!value || typeof value !== "object") return undefined;
  const pricing = value as Record<string, unknown>;
  const prompt =
    typeof pricing.prompt === "string" ? pricing.prompt : undefined;
  const completion =
    typeof pricing.completion === "string" ? pricing.completion : undefined;
  return prompt === undefined && completion === undefined
    ? undefined
    : { prompt, completion };
}

function parseCatalogEntry(
  entry: Record<string, unknown>,
): OpenRouterCatalogModel {
  const architecture = entry.architecture as
    | Record<string, unknown>
    | undefined;
  const topProvider = entry.top_provider as Record<string, unknown> | undefined;
  return {
    id: entry.id as string,
    supportedParameters: readStringArray(entry.supported_parameters),
    inputModalities: readStringArray(architecture?.input_modalities),
    outputModalities: readStringArray(architecture?.output_modalities),
    contextLength:
      readNumber(entry.context_length) ??
      readNumber(topProvider?.context_length),
    maxOutputTokens: readNumber(topProvider?.max_completion_tokens),
    pricing: readPricing(entry.pricing),
  };
}

/** Public provider metadata only; never send backend or LLM credentials. */
export async function fetchOpenRouterModels(): Promise<
  OpenRouterCatalogModel[]
> {
  const response = await fetch(OPENROUTER_MODELS_URL, {
    credentials: "omit",
    referrerPolicy: "no-referrer",
    signal: AbortSignal.timeout(CATALOG_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`OpenRouter model catalog returned ${response.status}`);
  }
  const catalog: unknown = await response.json();
  if (
    !catalog ||
    typeof catalog !== "object" ||
    !("data" in catalog) ||
    !Array.isArray(catalog.data) ||
    catalog.data.length === 0 ||
    !catalog.data.every(
      (model: unknown) =>
        model !== null &&
        typeof model === "object" &&
        "id" in model &&
        typeof model.id === "string" &&
        model.id.trim().length > 0,
    )
  ) {
    throw new Error("OpenRouter returned an invalid model catalog");
  }
  // Omitting pagination parameters requests the full catalog. IDs, including
  // vendor namespaces and catalog variants, are preserved verbatim. Dedupe by
  // id (last entry's metadata wins) while keeping first-seen order.
  const byId = new Map<string, OpenRouterCatalogModel>();
  for (const raw of catalog.data as Record<string, unknown>[]) {
    const parsed = parseCatalogEntry(raw);
    byId.set(parsed.id, parsed);
  }
  return [...byId.values()];
}
