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
 * Common resolver for schema field text (labels and descriptions).
 * Uses the i18n fallback chain:
 * 1. If schema provides an explicit translation key (contains $), use it directly
 * 2. Try the conventional key (SCHEMA$<PATH>$<ATTRIBUTE>)
 * 3. Fall back to the schema-provided value (untranslated)
 *
 * Logs a warning if no translation is found and falling back to schema value.
 */
function resolveSchemaFieldText(
  t: TFunction,
  fieldKey: string,
  attribute: "LABEL" | "DESCRIPTION" | "SECTION_LABEL",
  schemaValue: string | null | undefined,
): string | null {
  // If schema already provides a translation key, use it
  if (looksLikeTranslationKey(schemaValue)) {
    return t(schemaValue as string);
  }

  // Try conventional key
  const conventionalKey = toSchemaTranslationKey(fieldKey, attribute);
  const translated = t(conventionalKey, { defaultValue: "" });

  // If we got a translation, use it
  if (translated) {
    return translated;
  }

  // Log warning when falling back to untranslated schema value
  if (schemaValue) {
    // eslint-disable-next-line no-console
    console.warn(
      `[i18n] Missing translation for key "${conventionalKey}", falling back to: "${schemaValue}"`,
    );
  }

  return schemaValue ?? null;
}

/**
 * Resolves a field label using the i18n fallback chain.
 * @see resolveSchemaFieldText for the fallback chain details.
 */
export function resolveSchemaFieldLabel(
  t: TFunction,
  fieldKey: string,
  schemaValue: string,
): string {
  return resolveSchemaFieldText(t, fieldKey, "LABEL", schemaValue) ?? "";
}

/**
 * Resolves a field description using the i18n fallback chain.
 * @see resolveSchemaFieldText for the fallback chain details.
 */
export function resolveSchemaFieldDescription(
  t: TFunction,
  fieldKey: string,
  schemaValue?: string | null,
): string | null {
  return resolveSchemaFieldText(t, fieldKey, "DESCRIPTION", schemaValue);
}

/**
 * Resolves a section label using the i18n fallback chain.
 * @see resolveSchemaFieldText for the fallback chain details.
 */
export function resolveSchemaFieldSectionLabel(
  t: TFunction,
  sectionKey: string,
  schemaValue: string,
): string {
  return (
    resolveSchemaFieldText(t, sectionKey, "SECTION_LABEL", schemaValue) ?? ""
  );
}

/**
 * Resolves a choice label for select fields using the i18n fallback chain.
 * Convention: SCHEMA$<FIELD_PATH>$CHOICE$<CHOICE_VALUE>
 *
 * Logs a warning if no translation is found and falling back to schema label.
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
  const translated = t(conventionalKey, { defaultValue: "" });

  if (translated) {
    return translated;
  }

  // Log warning when falling back to untranslated schema label
  // eslint-disable-next-line no-console
  console.warn(
    `[i18n] Missing translation for key "${conventionalKey}", falling back to: "${schemaLabel}"`,
  );

  return schemaLabel;
}
