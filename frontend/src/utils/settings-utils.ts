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

/**
 * Parses and validates a max budget per task value.
 * Ensures the value is at least 1 dollar.
 * @param value - The string value to parse
 * @returns The parsed number if valid (>= 1), null otherwise
 */
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

  // Databricks-specific fields (only present when Databricks provider is selected)
  const databricksClientId =
    formData.get("databricks-client-id-input")?.toString() || undefined;
  const databricksClientSecret =
    formData.get("databricks-client-secret-input")?.toString() || undefined;
  const databricksU2mClientId =
    formData.get("databricks-u2m-client-id-input")?.toString() || undefined;
  const databricksU2mClientSecret =
    formData.get("databricks-u2m-client-secret-input")?.toString() || undefined;
  const databricksU2mRedirectUri =
    formData.get("databricks-u2m-redirect-uri-input")?.toString() || undefined;

  return {
    ...(Object.keys(agentSettings).length > 0
      ? { agent_settings_diff: agentSettings }
      : {}),
    ...(language ? { language } : {}),
    ...(databricksClientId !== undefined
      ? { databricks_client_id: databricksClientId }
      : {}),
    ...(databricksClientSecret !== undefined
      ? { databricks_client_secret: databricksClientSecret }
      : {}),
    ...(databricksU2mClientId !== undefined
      ? { databricks_u2m_client_id: databricksU2mClientId }
      : {}),
    ...(databricksU2mClientSecret !== undefined
      ? { databricks_u2m_client_secret: databricksU2mClientSecret }
      : {}),
    ...(databricksU2mRedirectUri !== undefined
      ? { databricks_u2m_redirect_uri: databricksU2mRedirectUri }
      : {}),
  };
};

/**
 * Checks if a settings page should be hidden based on feature flags.
 * Used by both the route loader and navigation hook to keep logic in sync.
 */
export function isSettingsPageHidden(
  path: string,
  featureFlags: WebClientFeatureFlags | undefined,
): boolean {
  if (
    featureFlags?.hide_llm_settings &&
    (path === "/settings" || path.startsWith("/settings/org-defaults"))
  )
    return true;
  if (featureFlags?.hide_users_page && path === "/settings/user") return true;
  if (featureFlags?.hide_billing_page && path === "/settings/billing")
    return true;
  if (featureFlags?.hide_integrations_page && path === "/settings/integrations")
    return true;
  return false;
}

/**
 * Find the first available settings page that is not hidden.
 * Returns null if no page is available (shouldn't happen in practice).
 */
export function getFirstAvailablePath(
  isSaas: boolean,
  featureFlags: WebClientFeatureFlags | undefined,
): string | null {
  const saasFallbackOrder = [
    { path: "/settings/user", hidden: !!featureFlags?.hide_users_page },
    {
      path: "/settings/integrations",
      hidden: !!featureFlags?.hide_integrations_page,
    },
    { path: "/settings/app", hidden: false },
    { path: "/settings", hidden: !!featureFlags?.hide_llm_settings },
    { path: "/settings/billing", hidden: !!featureFlags?.hide_billing_page },
    { path: "/settings/secrets", hidden: false },
    { path: "/settings/api-keys", hidden: false },
    { path: "/settings/mcp", hidden: false },
  ];

  const ossFallbackOrder = [
    { path: "/settings", hidden: !!featureFlags?.hide_llm_settings },
    { path: "/settings/mcp", hidden: false },
    {
      path: "/settings/integrations",
      hidden: !!featureFlags?.hide_integrations_page,
    },
    { path: "/settings/app", hidden: false },
    { path: "/settings/secrets", hidden: false },
  ];

  const fallbackOrder = isSaas ? saasFallbackOrder : ossFallbackOrder;
  const firstAvailable = fallbackOrder.find((item) => !item.hidden);

  return firstAvailable?.path ?? null;
}
