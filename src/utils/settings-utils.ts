import { WebClientFeatureFlags } from "#/api/option-service/option.types";
import { Settings, SettingsValue } from "#/types/settings";
import { getProviderId } from "#/utils/map-provider";
import type { SettingsContext } from "#/settings/registry";
import {
  getSettingsNavEntries,
  getRegisteredSettingsNavPaths,
} from "#/settings/nav-registry";
// Ensure the built-in OSS pages are registered before these helpers read the
// nav registry (they can run outside any React tree, e.g. in route loaders).
import "#/settings/register-settings-nav";

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

// These route helpers run outside React (route loaders, redirects), so they
// take feature flags directly rather than reading the React
// `useSettingsContext`. They build the same fact set the hook does, defaulting
// the facts a loader cannot know (backend kind / org) — no built-in page gates
// on those, so the defaults never change the result.
const featureFlagsToContext = (
  featureFlags: WebClientFeatureFlags | undefined,
): SettingsContext => ({
  backendKind: "local",
  orgId: null,
  featureFlags,
});

export function isSettingsPageHidden(
  path: string,
  featureFlags: WebClientFeatureFlags | undefined,
): boolean {
  const context = featureFlagsToContext(featureFlags);
  const visible = new Set(
    getSettingsNavEntries(context).map((entry) => entry.to),
  );
  const registered = new Set(getRegisteredSettingsNavPaths());
  // Only registered pages can be hidden; an unknown path is never "hidden".
  return registered.has(path) && !visible.has(path);
}

export function getFirstAvailablePath(
  featureFlags: WebClientFeatureFlags | undefined,
): string | null {
  // The first visible page in registry order. ``/settings/agents`` (the Agent
  // Profile library) is the lowest-ordered built-in and is always visible, so
  // it wins by default — every user lands where the agent is chosen, and the
  // LLM page is one nav-click away.
  const [firstVisible] = getSettingsNavEntries(
    featureFlagsToContext(featureFlags),
  );
  return firstVisible?.to ?? null;
}
