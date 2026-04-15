import type { TFunction } from "i18next";

export interface SettingsFieldConstraints {
  min?: number;
  max?: number;
  step?: number;
}

interface SettingsFieldMetadata {
  constraints?: SettingsFieldConstraints;
}

/**
 * Generates a conventional i18n translation key from a schema field key.
 *
 * Convention: SCHEMA$<SECTION>$<FIELD_NAME>$<ATTRIBUTE>
 * Examples:
 *   - "llm.api_key" + "LABEL" → "SCHEMA$LLM$API_KEY$LABEL"
 *   - "agent" + "DESCRIPTION" → "SCHEMA$AGENT$DESCRIPTION"
 *   - "llm" + "SECTION_LABEL" → "SCHEMA$LLM$SECTION_LABEL"
 *
 * This follows Rails-style i18n conventions where translation keys are
 * derived from model/attribute names using a predictable pattern.
 */
export function toSchemaTranslationKey(
  fieldKey: string,
  attribute: "LABEL" | "DESCRIPTION" | "SECTION_LABEL",
): string {
  const normalizedKey = fieldKey.replace(/\./g, "$").toUpperCase();
  return `SCHEMA$${normalizedKey}$${attribute}`;
}

const looksLikeTranslationKey = (value: string | null | undefined) =>
  Boolean(value?.includes("$"));

/**
 * Field-specific constraints (min/max/step for numeric inputs).
 * Labels and descriptions are now handled via convention-based i18n keys.
 */
const FIELD_METADATA: Record<string, SettingsFieldMetadata> = {
  "llm.top_p": {
    constraints: {
      min: 0,
      max: 1,
      step: 0.01,
    },
  },
  "llm.temperature": {
    constraints: {
      min: 0,
      max: 2,
      step: 0.1,
    },
  },
};

export function getSettingsFieldConstraints(fieldKey: string) {
  return FIELD_METADATA[fieldKey]?.constraints;
}

/**
 * Resolves a field label using the i18n fallback chain:
 * 1. If schema provides an explicit translation key (contains $), use it directly
 * 2. Try the conventional key (SCHEMA$<PATH>$LABEL)
 * 3. Fall back to the schema-provided value (untranslated)
 */
export function resolveSchemaFieldLabel(
  t: TFunction,
  fieldKey: string,
  schemaValue: string,
): string {
  // If schema already provides a translation key, use it
  if (looksLikeTranslationKey(schemaValue)) {
    return t(schemaValue);
  }

  // Try conventional key, fall back to schema value
  const conventionalKey = toSchemaTranslationKey(fieldKey, "LABEL");
  return t(conventionalKey, { defaultValue: schemaValue });
}

/**
 * Resolves a field description using the i18n fallback chain:
 * 1. If schema provides an explicit translation key (contains $), use it directly
 * 2. Try the conventional key (SCHEMA$<PATH>$DESCRIPTION)
 * 3. Fall back to the schema-provided value (untranslated), or null if not provided
 */
export function resolveSchemaFieldDescription(
  t: TFunction,
  fieldKey: string,
  schemaValue?: string | null,
): string | null {
  // If schema already provides a translation key, use it
  if (looksLikeTranslationKey(schemaValue)) {
    // TypeScript needs assurance that schemaValue is a string here
    return t(schemaValue as string);
  }

  // Try conventional key
  const conventionalKey = toSchemaTranslationKey(fieldKey, "DESCRIPTION");
  const translated = t(conventionalKey, { defaultValue: "" });

  // If we got a translation, use it; otherwise fall back to schema value
  if (translated) {
    return translated;
  }

  return schemaValue ?? null;
}

/**
 * Resolves a section label using the i18n fallback chain:
 * 1. If schema provides an explicit translation key (contains $), use it directly
 * 2. Try the conventional key (SCHEMA$<SECTION>$SECTION_LABEL)
 * 3. Fall back to the schema-provided value (untranslated)
 */
export function resolveSchemaFieldSectionLabel(
  t: TFunction,
  sectionKey: string,
  schemaValue: string,
): string {
  // If schema already provides a translation key, use it
  if (looksLikeTranslationKey(schemaValue)) {
    return t(schemaValue);
  }

  // Try conventional key, fall back to schema value
  const conventionalKey = toSchemaTranslationKey(sectionKey, "SECTION_LABEL");
  return t(conventionalKey, { defaultValue: schemaValue });
}

/**
 * Resolves a choice label for select fields using the i18n fallback chain.
 * Convention: SCHEMA$<FIELD_PATH>$CHOICE$<CHOICE_VALUE>
 */
export function resolveSchemaChoiceLabel(
  t: TFunction,
  fieldKey: string,
  choiceValue: string | number | boolean,
  schemaLabel: string,
): string {
  if (looksLikeTranslationKey(schemaLabel)) {
    return t(schemaLabel);
  }

  const normalizedFieldKey = fieldKey.replace(/\./g, "$").toUpperCase();
  const normalizedChoiceValue = String(choiceValue).toUpperCase();
  const conventionalKey = `SCHEMA$${normalizedFieldKey}$CHOICE$${normalizedChoiceValue}`;
  return t(conventionalKey, { defaultValue: schemaLabel });
}
