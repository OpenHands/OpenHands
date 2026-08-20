import type { SettingsSchema } from "#/types/settings";
import { LLM_SUBSCRIPTION_SCHEMA_FIELDS } from "#/constants/llm-subscription";

const LLM_SECTION_KEY = "llm";

export function withLlmSubscriptionSchemaFields(
  schema: SettingsSchema | null | undefined,
): SettingsSchema | null | undefined {
  if (!schema?.sections) return schema;

  let changed = false;
  const sections = schema.sections.map((section) => {
    if (section.key !== LLM_SECTION_KEY) return section;

    const existingKeys = new Set(section.fields.map((field) => field.key));
    const missingFields = LLM_SUBSCRIPTION_SCHEMA_FIELDS.filter(
      (field) => !existingKeys.has(field.key),
    );

    if (missingFields.length === 0) return section;
    changed = true;
    return {
      ...section,
      fields: [...section.fields, ...missingFields],
    };
  });

  if (!changed) return schema;
  return { ...schema, sections };
}
