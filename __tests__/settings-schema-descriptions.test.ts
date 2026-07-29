import { describe, expect, it } from "vitest";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";

describe("settings schema descriptions", () => {
  it("provides helper descriptions for every schema-driven settings field", () => {
    const schemas = [
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema,
      MOCK_DEFAULT_USER_SETTINGS.conversation_settings_schema,
    ].filter((schema): schema is NonNullable<typeof schema> => Boolean(schema));

    const missingDescriptions = schemas.flatMap((schema) =>
      schema.sections.flatMap((section) =>
        section.fields
          .filter((field) => !field.description?.trim())
          .map((field) => field.key),
      ),
    );

    expect(missingDescriptions).toEqual([]);
  });
});
