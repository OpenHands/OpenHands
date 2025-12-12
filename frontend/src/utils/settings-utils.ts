import { Settings } from "#/types/settings";
import { getProviderId } from "#/utils/map-provider";

const extractBasicFormData = (formData: FormData) => {
  const providerDisplay = formData.get("llm-provider-input")?.toString();
  const provider = providerDisplay ? getProviderId(providerDisplay) : undefined;
  const model = formData.get("llm-model-input")?.toString();

  const LLM_MODEL = `${provider}/${model}`;
  const LLM_API_KEY = formData.get("llm-api-key-input")?.toString();
  const AGENT = formData.get("agent")?.toString();
  const LANGUAGE = formData.get("language")?.toString();

  return {
    LLM_MODEL,
    LLM_API_KEY,
    AGENT,
    LANGUAGE,
  };
};

const extractAdvancedFormData = (formData: FormData) => {
  const keys = Array.from(formData.keys());
  const isUsingAdvancedOptions = keys.includes("use-advanced-options");

  let CUSTOM_LLM_MODEL: string | undefined;
  let LLM_BASE_URL: string | undefined;
  let CONFIRMATION_MODE = false;
  let SECURITY_ANALYZER: string | undefined;
  let ENABLE_DEFAULT_CONDENSER = true;

  if (isUsingAdvancedOptions) {
    CUSTOM_LLM_MODEL = formData.get("custom-model")?.toString();
    LLM_BASE_URL = formData.get("base-url")?.toString();
    CONFIRMATION_MODE = keys.includes("confirmation-mode");
    if (CONFIRMATION_MODE) {
      // only set securityAnalyzer if confirmationMode is enabled
      SECURITY_ANALYZER = formData.get("security-analyzer")?.toString();
    }
    ENABLE_DEFAULT_CONDENSER = keys.includes("enable-default-condenser");
  }

  return {
    CUSTOM_LLM_MODEL,
    LLM_BASE_URL,
    CONFIRMATION_MODE,
    SECURITY_ANALYZER,
    ENABLE_DEFAULT_CONDENSER,
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
  // Ensure the value is at least 1 dollar and is a finite number
  return parsedValue && parsedValue >= 1 && Number.isFinite(parsedValue)
    ? parsedValue
    : null;
};

export const extractSettings = (formData: FormData): Partial<Settings> => {
  const { LLM_MODEL, LLM_API_KEY, AGENT, LANGUAGE } =
    extractBasicFormData(formData);

  const {
    CUSTOM_LLM_MODEL,
    LLM_BASE_URL,
    CONFIRMATION_MODE,
    SECURITY_ANALYZER,
    ENABLE_DEFAULT_CONDENSER,
  } = extractAdvancedFormData(formData);

  return {
    llm_model: CUSTOM_LLM_MODEL || LLM_MODEL,
    llm_api_key_set: !!LLM_API_KEY,
    agent: AGENT,
    language: LANGUAGE,
    llm_base_url: LLM_BASE_URL,
    confirmation_mode: CONFIRMATION_MODE,
    security_analyzer: SECURITY_ANALYZER,
    enable_default_condenser: ENABLE_DEFAULT_CONDENSER,
    llm_api_key: LLM_API_KEY,
  };
};
