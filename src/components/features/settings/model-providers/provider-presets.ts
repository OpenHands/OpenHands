import type { WireApi } from "#/api/model-providers-service";

/**
 * Preset provider kinds surfaced in the "Add provider" picker (issue #15492
 * wireframe). Each preset seeds sensible defaults into the provider form; the
 * user can still override them. `custom` is the escape hatch for any
 * OpenAI-compatible endpoint.
 */
export interface ProviderPreset {
  kind: string;
  label: string;
  description: string;
  defaultBaseUrl?: string;
  defaultWireApi?: WireApi;
}

export const PROVIDER_PRESETS: ProviderPreset[] = [
  {
    kind: "custom",
    label: "Custom endpoint",
    description:
      "Any OpenAI-compatible HTTP endpoint (vLLM, OpenRouter, fine-tune, etc.).",
    defaultWireApi: "chat",
  },
  {
    kind: "openai",
    label: "OpenAI",
    description: "GPT models over the Responses or Chat Completions API.",
    defaultBaseUrl: "https://api.openai.com/v1",
    defaultWireApi: "auto",
  },
  {
    kind: "anthropic",
    label: "Anthropic",
    description: "Hosted Claude models over the Messages API.",
    defaultWireApi: "chat",
  },
  {
    kind: "azure",
    label: "Azure OpenAI",
    description:
      "Service deployments via your resource host, API version, and per-model deployment names.",
    defaultWireApi: "chat",
  },
  {
    kind: "foundry-local",
    label: "Foundry Local",
    description:
      "Local runtime with no API key, over an OpenAI-compatible endpoint.",
    defaultWireApi: "chat",
  },
  {
    kind: "microsoft-foundry",
    label: "Microsoft Foundry",
    description:
      "OpenAI-compatible endpoints. Defaults to the Chat Completions wire format.",
    defaultWireApi: "chat",
  },
];
