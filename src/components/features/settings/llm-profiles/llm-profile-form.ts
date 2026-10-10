import { LLM_PROVIDER_CONNECTION_KEY } from "#/constants/llm-provider-connection";
import {
  LLM_AUTH_TYPE_API_KEY,
  LLM_AUTH_TYPE_KEY,
  LLM_AUTH_TYPE_SUBSCRIPTION,
  LLM_SUBSCRIPTION_VENDOR_KEY,
  OPENAI_SUBSCRIPTION_VENDOR,
  resolveLlmAuthType,
} from "#/constants/llm-subscription";
import type { SettingsSchema } from "#/types/settings";
import { isOpenHandsProviderModel } from "#/utils/format-model-name";
import {
  normalizeFieldValue,
  type SettingsFormValues,
  type SettingsView,
} from "#/utils/sdk-settings-schema";

/**
 * Form values for an LLM profile, so every tab of the embedded LLM form shows
 * the profile's real values rather than the active settings. `config` is the
 * profile's flat LLM config (`{ model, api_key, base_url, ... }`, with
 * `api_key` encrypted); an empty config yields the schema defaults. Fields
 * absent from the schema are not form values: pass `config` as the
 * `baseConfig` of {@link buildProfileLlmConfig} to keep them on save.
 */
export function profileConfigToFormValues(
  schema: SettingsSchema | null | undefined,
  config: Record<string, unknown>,
): SettingsFormValues {
  const llmFields =
    schema?.sections.find((section) => section.key === "llm")?.fields ?? [];
  const values: SettingsFormValues = {};
  for (const field of llmFields) {
    const flatKey = field.key.startsWith("llm.")
      ? field.key.slice("llm.".length)
      : field.key;
    values[field.key] = normalizeFieldValue(field, config[flatKey]);
  }

  // Safety net for the specially-rendered keys when the schema is
  // unavailable, so the form still works without it.
  if (llmFields.length === 0) {
    values["llm.model"] = (config.model as string) ?? "";
    values["llm.api_key"] = (config.api_key as string) ?? "";
    values["llm.base_url"] = (config.base_url as string) ?? "";
    values[LLM_AUTH_TYPE_KEY] = resolveLlmAuthType(config.auth_type);
    values[LLM_SUBSCRIPTION_VENDOR_KEY] =
      (config.subscription_vendor as string) ?? OPENAI_SUBSCRIPTION_VENDOR;
  }

  // Seed the provider-connection link explicitly (it is excluded from the
  // schema-driven inputs), so an unchanged profile keeps its connection.
  values[LLM_PROVIDER_CONNECTION_KEY] =
    typeof config.provider_connection_id === "string"
      ? config.provider_connection_id
      : "";

  return values;
}

export interface BuildProfileLlmConfigOptions {
  /**
   * The full LLM config of the profile the form was filled from, or `{}` for a
   * new profile. Merging the form's changes over it preserves fields the user
   * did not touch, including ones hidden in the current tab or absent from the
   * schema; the backend replaces the whole LLM object, so a partial config
   * would otherwise reset everything else to defaults.
   */
  baseConfig: Record<string, unknown>;
  /** The form's coerced, dirty-only `llm` payload (`getDirtyPayload().llm`). */
  dirtyLlm: Record<string, unknown>;
  /** The form's current values (`SdkSectionSaveControl.values`). */
  values: SettingsFormValues;
  /** The form's current view (`SdkSectionSaveControl.view`). */
  view: SettingsView;
  /** Whether this backend can link a profile to a provider connection. */
  supportsConnections: boolean;
  isCloud: boolean;
  /**
   * The loaded subscription models. When given, a subscription profile whose
   * model is not one of them gets the first, which is the model the form's
   * subscription picker shows in that case.
   */
  subscriptionModels?: readonly string[];
}

/**
 * The complete LLM config to save as a profile from an embedded LLM form, and
 * the provider connection it links to (`""` when none). The same rules apply
 * wherever a form edits a profile: the LLM profile editor and onboarding.
 */
export function buildProfileLlmConfig({
  baseConfig,
  dirtyLlm,
  values,
  view,
  supportsConnections,
  isCloud,
  subscriptionModels,
}: BuildProfileLlmConfigOptions): {
  llmConfig: Record<string, unknown>;
  connectionId: string;
} {
  const base = { ...baseConfig };
  const didChangeModelInBasic =
    view === "basic" &&
    Object.prototype.hasOwnProperty.call(dirtyLlm, "model") &&
    dirtyLlm.model !== base.model;
  const llmConfig: Record<string, unknown> = { ...base, ...dirtyLlm };
  const authType = resolveLlmAuthType(llmConfig.auth_type);

  // A profile linked to a provider connection sources its credential from the
  // connection, so it never carries an inline api_key / base_url. The form
  // value is the source of truth: empty (or absent) means "not linked".
  const connectionId = supportsConnections
    ? String(values[LLM_PROVIDER_CONNECTION_KEY] ?? "").trim()
    : "";

  if (authType === LLM_AUTH_TYPE_SUBSCRIPTION) {
    const model = typeof llmConfig.model === "string" ? llmConfig.model : "";
    if (subscriptionModels?.length && !subscriptionModels.includes(model)) {
      [llmConfig.model] = subscriptionModels;
    }
    llmConfig.auth_type = LLM_AUTH_TYPE_SUBSCRIPTION;
    llmConfig.subscription_vendor = OPENAI_SUBSCRIPTION_VENDOR;
    llmConfig.provider_connection_id = null;
    delete llmConfig.api_key;
    delete llmConfig.base_url;
  } else if (connectionId) {
    llmConfig.auth_type = LLM_AUTH_TYPE_API_KEY;
    llmConfig.subscription_vendor = null;
    llmConfig.provider_connection_id = connectionId;
    delete llmConfig.api_key;
    delete llmConfig.base_url;
  } else {
    llmConfig.auth_type = LLM_AUTH_TYPE_API_KEY;
    llmConfig.subscription_vendor = null;
    // Clear any prior link so unlinking sticks. Only relevant where provider
    // connections exist; otherwise the field stays untouched below.
    if (supportsConnections) llmConfig.provider_connection_id = null;

    // On cloud the OpenHands provider is backed by a server-minted LLM key,
    // so the profile must not carry an inline api_key / base_url — let the
    // backend attach its own credential when the profile is saved.
    const isCloudOpenHandsProvider =
      isCloud &&
      isOpenHandsProviderModel(
        typeof llmConfig.model === "string" ? llmConfig.model : "",
      );
    if (isCloudOpenHandsProvider) {
      delete llmConfig.api_key;
      delete llmConfig.base_url;
    } else {
      // The Basic tab has no base_url field. Preserve an existing hidden value
      // when the model did not actually change; if the user chooses a new model,
      // drop the old base URL so provider defaults can apply to that model.
      if (didChangeModelInBasic) {
        delete llmConfig.base_url;
      }

      // API key handling: an empty value means "no change" (the UX doesn't
      // support clearing a key). Preserve the existing encrypted key from the
      // base profile, or omit api_key entirely for a new profile. A newly
      // typed key arrives in `dirtyLlm` and wins.
      if (
        typeof llmConfig.api_key !== "string" ||
        llmConfig.api_key.trim() === ""
      ) {
        const existingKey =
          typeof base.api_key === "string" ? base.api_key : "";
        if (existingKey) {
          llmConfig.api_key = existingKey;
        } else {
          delete llmConfig.api_key;
        }
      }
    }
  }

  return { llmConfig, connectionId };
}
