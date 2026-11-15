import type { SettingsSchema, SettingsValue } from "#/types/settings";
import {
  coerceFieldValue,
  normalizeFieldValue,
  type SettingsFormValues,
} from "#/utils/sdk-settings-schema";

/** Agent-settings sections an OpenHands agent profile stores as its own. */
export type ProfileBehaviorSection = "condenser" | "verification";

/**
 * Fields of those sections that stay global: every agent shares the critic API
 * key, and the confirmation controls belong to conversation settings.
 */
export const GLOBAL_BEHAVIOR_FIELD_KEYS = new Set([
  "verification.critic_api_key",
  "verification.confirmation_mode",
  "verification.security_analyzer",
]);

function profileFields(
  schema: SettingsSchema | null | undefined,
  section: ProfileBehaviorSection,
) {
  return (
    schema?.sections
      .find((candidate) => candidate.key === section)
      ?.fields.filter((field) => !GLOBAL_BEHAVIOR_FIELD_KEYS.has(field.key)) ??
    []
  );
}

/** Keys of the section's fields each profile stores. */
export function profileBehaviorFieldKeys(
  schema: SettingsSchema | null | undefined,
  section: ProfileBehaviorSection,
): string[] {
  return profileFields(schema, section).map((field) => field.key);
}

/** A stored profile section as form values; fields it lacks take the schema default. */
export function toBehaviorFormValues(
  schema: SettingsSchema | null | undefined,
  section: ProfileBehaviorSection,
  stored: Record<string, unknown>,
): SettingsFormValues {
  const prefix = `${section}.`;
  return Object.fromEntries(
    profileFields(schema, section).map((field) => [
      field.key,
      normalizeFieldValue(field, stored[field.key.slice(prefix.length)]),
    ]),
  );
}

/** Form values as the profile section to store. Throws on a value that fails coercion. */
export function fromBehaviorFormValues(
  schema: SettingsSchema | null | undefined,
  section: ProfileBehaviorSection,
  values: SettingsFormValues,
): Record<string, SettingsValue> {
  const prefix = `${section}.`;
  return Object.fromEntries(
    profileFields(schema, section)
      .filter((field) => field.key in values)
      .map((field) => [
        field.key.slice(prefix.length),
        coerceFieldValue(field, values[field.key]),
      ]),
  );
}
