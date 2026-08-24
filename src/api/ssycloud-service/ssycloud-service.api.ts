import type { LLMModel } from "#/api/config-service/config-service.types";
import {
  SSYCLOUD_MODELS_URL,
  SSYCLOUD_PROVIDER_ID,
} from "#/constants/ssycloud";

interface SSYCloudModelResponseItem {
  id?: unknown;
  support_apis?: unknown;
}

interface SSYCloudModelsResponse {
  data?: unknown;
}

const supportsChatCompletions = (supportApis: unknown): boolean => {
  if (!Array.isArray(supportApis)) return true;
  return supportApis.some(
    (api) =>
      typeof api === "string" &&
      (api === "/v1/chat/completions" || api === "/chat/completions"),
  );
};

export async function fetchSSYCloudModels(
  apiKey: string,
  signal?: AbortSignal,
): Promise<LLMModel[]> {
  const trimmedApiKey = apiKey.trim();
  if (!trimmedApiKey) return [];

  const response = await fetch(SSYCLOUD_MODELS_URL, {
    method: "GET",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${trimmedApiKey}`,
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(`SSYCloud models request failed (${response.status})`);
  }

  const payload = (await response.json()) as SSYCloudModelsResponse;
  if (!Array.isArray(payload.data)) {
    throw new Error("SSYCloud models response is missing data");
  }

  const modelIds = new Set<string>();
  for (const rawItem of payload.data) {
    if (!rawItem || typeof rawItem !== "object") continue;
    const item = rawItem as SSYCloudModelResponseItem;
    if (
      typeof item.id === "string" &&
      item.id.trim().length > 0 &&
      supportsChatCompletions(item.support_apis)
    ) {
      modelIds.add(item.id.trim());
    }
  }

  return [...modelIds]
    .sort((left, right) => left.localeCompare(right))
    .map((name) => ({
      provider: SSYCLOUD_PROVIDER_ID,
      name,
      verified: false,
    }));
}
