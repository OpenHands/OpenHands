import type { SettingsChoice, SettingsFieldSchema } from "#/types/settings";

export const LLM_AUTH_TYPE_KEY = "llm.auth_type";
export const LLM_SUBSCRIPTION_VENDOR_KEY = "llm.subscription_vendor";
export const LLM_AUTH_TYPE_API_KEY = "api_key";
export const LLM_AUTH_TYPE_SUBSCRIPTION = "subscription";
export const OPENAI_SUBSCRIPTION_VENDOR = "openai";

export const OPENAI_SUBSCRIPTION_MODELS_PATH =
  "/api/llm/subscription/openai/models";
export const OPENAI_SUBSCRIPTION_STATUS_PATH =
  "/api/llm/subscription/openai/status";
export const OPENAI_SUBSCRIPTION_DEVICE_START_PATH =
  "/api/llm/subscription/openai/device/start";
export const OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH =
  "/api/llm/subscription/openai/device/poll";
export const OPENAI_SUBSCRIPTION_LOGOUT_PATH =
  "/api/llm/subscription/openai/logout";

export type LlmAuthType =
  | typeof LLM_AUTH_TYPE_API_KEY
  | typeof LLM_AUTH_TYPE_SUBSCRIPTION;

export const LLM_AUTH_TYPE_CHOICES: SettingsChoice[] = [
  { label: "API key", value: LLM_AUTH_TYPE_API_KEY },
  { label: "ChatGPT subscription", value: LLM_AUTH_TYPE_SUBSCRIPTION },
];

const LLM_AUTH_TYPE_FIELD: SettingsFieldSchema = {
  key: LLM_AUTH_TYPE_KEY,
  label: "Authentication",
  description:
    "Choose whether this profile uses API credentials or a ChatGPT subscription.",
  section: "llm",
  section_label: "LLM",
  value_type: "string",
  default: LLM_AUTH_TYPE_API_KEY,
  choices: LLM_AUTH_TYPE_CHOICES,
  depends_on: [],
  prominence: "critical",
  secret: false,
  required: true,
};

const LLM_SUBSCRIPTION_VENDOR_FIELD: SettingsFieldSchema = {
  key: LLM_SUBSCRIPTION_VENDOR_KEY,
  label: "Subscription provider",
  description: "Provider used for subscription-backed LLM access.",
  section: "llm",
  section_label: "LLM",
  value_type: "string",
  default: OPENAI_SUBSCRIPTION_VENDOR,
  choices: [{ label: "OpenAI", value: OPENAI_SUBSCRIPTION_VENDOR }],
  depends_on: [],
  prominence: "critical",
  secret: false,
  required: true,
};

export const LLM_SUBSCRIPTION_SCHEMA_FIELDS = [
  LLM_AUTH_TYPE_FIELD,
  LLM_SUBSCRIPTION_VENDOR_FIELD,
];

export function resolveLlmAuthType(value: unknown): LlmAuthType {
  return value === LLM_AUTH_TYPE_SUBSCRIPTION
    ? LLM_AUTH_TYPE_SUBSCRIPTION
    : LLM_AUTH_TYPE_API_KEY;
}

export function isSubscriptionLlmConfig(
  llm: Record<string, unknown> | null | undefined,
): boolean {
  return resolveLlmAuthType(llm?.auth_type) === LLM_AUTH_TYPE_SUBSCRIPTION;
}
