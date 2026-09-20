export const OPENROUTER_PROVIDER = "openrouter";
const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
const CATALOG_TIMEOUT_MS = 5000;

/** Public provider metadata only; never send backend or LLM credentials. */
export async function fetchOpenRouterModels(): Promise<string[]> {
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
  // vendor namespaces and catalog variants, are preserved verbatim.
  return [...new Set(catalog.data.map((model: { id: string }) => model.id))];
}
