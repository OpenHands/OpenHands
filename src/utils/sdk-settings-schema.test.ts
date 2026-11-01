import { describe, it, expect } from "vitest";
import type { SettingsFieldSchema } from "#/types/settings";
import { coerceFieldValue } from "./sdk-settings-schema";

function makeField(
  overrides: Partial<SettingsFieldSchema> = {},
): SettingsFieldSchema {
  return {
    key: "llm.base_url",
    label: "Base URL",
    section: "llm",
    section_label: "LLM",
    value_type: "string",
    choices: [],
    depends_on: [],
    prominence: "major",
    secret: false,
    required: false,
    ...overrides,
  };
}

describe("coerceFieldValue", () => {
  describe("llm.base_url validation (#15778)", () => {
    it("accepts a valid https URL", () => {
      expect(
        coerceFieldValue(
          makeField(),
          "https://api.openai.com/v1",
        ),
      ).toBe("https://api.openai.com/v1");
    });

    it("accepts a valid http URL", () => {
      expect(
        coerceFieldValue(makeField(), "http://localhost:11434"),
      ).toBe("http://localhost:11434");
    });

    it("treats an empty value as unset", () => {
      expect(coerceFieldValue(makeField(), "")).toBeNull();
    });

    it("rejects a non-URL string", () => {
      expect(() =>
        coerceFieldValue(makeField(), "not-a-url"),
      ).toThrow(/valid http\(s\) URL/);
    });

    it("rejects a URL with an unsupported protocol", () => {
      expect(() =>
        coerceFieldValue(makeField(), "ftp://example.com"),
      ).toThrow(/valid http\(s\) URL/);
    });
  });
});
