import { WebClientFeatureFlags } from "#/api/option-service/option.types";
import { Settings, SettingsValue } from "#/types/settings";
import { getProviderId } from "#/utils/map-provider";

const extractBasicFormData = (formData: FormData) => {
  const providerDisplay = formData.get("llm-provider-input")?.toString();
  const provider = providerDisplay ? getProviderId(providerDisplay) : undefined;
  const model = formData.get("llm-model-input")?.toString();

  return {
    llmModel: provider && model ? `${provider}/${model}` : undefined,
    llmApiKey: formData.get("llm-api-key-input")?.toString(),
    agent: formData.get("agent")?.toString(),
    language: formData.get("language")?.toString(),
  };
};

export const parseMaxBudgetPerTask = (value: string): number | null => {
  if (!value) {
    return null;
  }

  const parsedValue = parseFloat(value);
  return parsedValue && parsedValue >= 1 && Number.isFinite(parsedValue)
    ? parsedValue
    : null;
};

export const extractSettings = (
  formData: FormData,
): Partial<Settings> & Record<string, unknown> => {
  const { llmModel, llmApiKey, agent, language } =
    extractBasicFormData(formData);

  const llm: Record<string, unknown> = {};
  if (llmModel) llm.model = llmModel;
  if (llmApiKey !== undefined) llm.api_key = llmApiKey;

  const agentSettings: Record<string, SettingsValue> = {};
  if (Object.keys(llm).length > 0)
    agentSettings.llm = llm as Record<string, SettingsValue>;
  if (agent) agentSettings.agent = agent;

  return {
    ...(Object.keys(agentSettings).length > 0
      ? { agent_settings_diff: agentSettings }
      : {}),
    ...(language ? { language } : {}),
  };
};

export function isSettingsPageHidden(
  path: string,
  featureFlags: WebClientFeatureFlags | undefined,
): boolean {
  if (featureFlags?.hide_llm_settings && path === "/settings/llm") return true;
  return false;
}

export function getFirstAvailablePath(
  featureFlags: WebClientFeatureFlags | undefined,
): string | null {
  // ``/settings/agents`` (the Agent Profile library — the "Agent" page) always
  // wins: it is where the agent is defined (OpenHands / ACP, via the active
  // profile) and is always available regardless of feature flags. Landing here
  // keeps routing simple — every user lands where the agent is chosen, and
  // the LLM page is one nav-click away.
  const fallbackOrder = [
    { path: "/settings/agents", hidden: false },
    { path: "/settings/llm", hidden: !!featureFlags?.hide_llm_settings },
    { path: "/settings", hidden: !!featureFlags?.hide_llm_settings },
    { path: "/settings/app", hidden: false },
    { path: "/settings/secrets", hidden: false },
  ];

  const firstAvailable = fallbackOrder.find((item) => !item.hidden);
  return firstAvailable?.path ?? null;
}
