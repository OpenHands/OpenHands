import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { TFunction } from "i18next";
import {
  toSchemaTranslationKey,
  resolveSchemaFieldLabel,
  resolveSchemaFieldDescription,
  resolveSchemaFieldSectionLabel,
  resolveSchemaChoiceLabel,
  getSettingsFieldConstraints,
} from "./sdk-settings-field-metadata";

// Mock console.warn to suppress warnings during tests and allow verification
const originalWarn = console.warn;
let warnSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  warnSpy.mockRestore();
  console.warn = originalWarn;
});

describe("toSchemaTranslationKey", () => {
  it("generates correct key for simple field", () => {
    expect(toSchemaTranslationKey("agent", "LABEL")).toBe("SCHEMA$AGENT$LABEL");
  });

  it("generates correct key for nested field", () => {
    expect(toSchemaTranslationKey("llm.api_key", "LABEL")).toBe(
      "SCHEMA$LLM$API_KEY$LABEL",
    );
  });

  it("generates correct key for deeply nested field", () => {
    expect(toSchemaTranslationKey("llm.draft.model", "DESCRIPTION")).toBe(
      "SCHEMA$LLM$DRAFT$MODEL$DESCRIPTION",
    );
  });

  it("generates correct key for section label", () => {
    expect(toSchemaTranslationKey("llm", "SECTION_LABEL")).toBe(
      "SCHEMA$LLM$SECTION_LABEL",
    );
  });
});

describe("resolveSchemaFieldLabel", () => {
  it("returns translated value when conventional key exists", () => {
    const mockT = vi.fn((key: string, options?: { defaultValue: string }) => {
      if (key === "SCHEMA$LLM$API_KEY$LABEL") return "Translated API Key";
      return options?.defaultValue ?? key;
    }) as unknown as TFunction;

    const result = resolveSchemaFieldLabel(mockT, "llm.api_key", "API Key");
    expect(result).toBe("Translated API Key");
  });

  it("returns schema value when no translation exists", () => {
    const mockT = vi.fn(
      (key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? key,
    ) as unknown as TFunction;

    const result = resolveSchemaFieldLabel(
      mockT,
      "unknown.field",
      "Fallback Label",
    );
    expect(result).toBe("Fallback Label");
  });

  it("uses explicit translation key when provided in schema", () => {
    const mockT = vi.fn((key: string) => {
      if (key === "CUSTOM$KEY") return "Custom Translation";
      return key;
    }) as unknown as TFunction;

    const result = resolveSchemaFieldLabel(mockT, "some.field", "CUSTOM$KEY");
    expect(result).toBe("Custom Translation");
    expect(mockT).toHaveBeenCalledWith("CUSTOM$KEY");
  });
});

describe("resolveSchemaFieldDescription", () => {
  it("returns translated description when conventional key exists", () => {
    const mockT = vi.fn((key: string, options?: { defaultValue: string }) => {
      if (key === "SCHEMA$LLM$TEMPERATURE$DESCRIPTION")
        return "Translated description";
      return options?.defaultValue ?? "";
    }) as unknown as TFunction;

    const result = resolveSchemaFieldDescription(
      mockT,
      "llm.temperature",
      "Original desc",
    );
    expect(result).toBe("Translated description");
  });

  it("returns schema value when no translation exists", () => {
    const mockT = vi.fn(
      (_key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? "",
    ) as unknown as TFunction;

    const result = resolveSchemaFieldDescription(
      mockT,
      "unknown.field",
      "Fallback Description",
    );
    expect(result).toBe("Fallback Description");
  });

  it("returns null when no description provided and no translation", () => {
    const mockT = vi.fn(
      (_key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? "",
    ) as unknown as TFunction;

    const result = resolveSchemaFieldDescription(mockT, "some.field", null);
    expect(result).toBeNull();
  });

  it("uses explicit translation key when provided in schema", () => {
    const mockT = vi.fn((key: string) => {
      if (key === "CUSTOM$DESC") return "Custom Description";
      return "";
    }) as unknown as TFunction;

    const result = resolveSchemaFieldDescription(
      mockT,
      "some.field",
      "CUSTOM$DESC",
    );
    expect(result).toBe("Custom Description");
  });
});

describe("resolveSchemaFieldSectionLabel", () => {
  it("returns translated section label when conventional key exists", () => {
    const mockT = vi.fn((key: string, options?: { defaultValue: string }) => {
      if (key === "SCHEMA$LLM$SECTION_LABEL") return "Language Model";
      return options?.defaultValue ?? key;
    }) as unknown as TFunction;

    const result = resolveSchemaFieldSectionLabel(mockT, "llm", "LLM");
    expect(result).toBe("Language Model");
  });

  it("returns schema value when no translation exists", () => {
    const mockT = vi.fn(
      (key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? key,
    ) as unknown as TFunction;

    const result = resolveSchemaFieldSectionLabel(
      mockT,
      "custom_section",
      "Custom Section",
    );
    expect(result).toBe("Custom Section");
  });
});

describe("resolveSchemaChoiceLabel", () => {
  it("returns translated choice label when conventional key exists", () => {
    const mockT = vi.fn((key: string, options?: { defaultValue: string }) => {
      if (key === "SCHEMA$LLM$REASONING_EFFORT$CHOICE$HIGH") return "Hoch";
      return options?.defaultValue ?? key;
    }) as unknown as TFunction;

    const result = resolveSchemaChoiceLabel(
      mockT,
      "llm.reasoning_effort",
      "high",
      "High",
    );
    expect(result).toBe("Hoch");
  });

  it("returns schema value when no translation exists", () => {
    const mockT = vi.fn(
      (key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? key,
    ) as unknown as TFunction;

    const result = resolveSchemaChoiceLabel(
      mockT,
      "some.field",
      "option1",
      "Option 1",
    );
    expect(result).toBe("Option 1");
  });

  it("handles boolean choice values", () => {
    const mockT = vi.fn((key: string, options?: { defaultValue: string }) => {
      if (key === "SCHEMA$CONFIRMATION_MODE$CHOICE$TRUE") return "Enabled";
      return options?.defaultValue ?? key;
    }) as unknown as TFunction;

    const result = resolveSchemaChoiceLabel(
      mockT,
      "confirmation_mode",
      true,
      "Yes",
    );
    expect(result).toBe("Enabled");
  });

  it("handles numeric choice values", () => {
    const mockT = vi.fn((key: string, options?: { defaultValue: string }) => {
      if (key === "SCHEMA$PRIORITY$CHOICE$1") return "Low Priority";
      return options?.defaultValue ?? key;
    }) as unknown as TFunction;

    const result = resolveSchemaChoiceLabel(mockT, "priority", 1, "Low");
    expect(result).toBe("Low Priority");
  });

  it("uses explicit translation key when provided in schema", () => {
    const mockT = vi.fn((key: string) => {
      if (key === "CUSTOM$CHOICE$KEY") return "Custom Choice";
      return key;
    }) as unknown as TFunction;

    const result = resolveSchemaChoiceLabel(
      mockT,
      "some.field",
      "value",
      "CUSTOM$CHOICE$KEY",
    );
    expect(result).toBe("Custom Choice");
  });
});

describe("getSettingsFieldConstraints", () => {
  it("returns constraints for known fields", () => {
    const constraints = getSettingsFieldConstraints("llm.top_p");
    expect(constraints).toEqual({
      min: 0,
      max: 1,
      step: 0.01,
    });
  });

  it("returns undefined for unknown fields", () => {
    const constraints = getSettingsFieldConstraints("unknown.field");
    expect(constraints).toBeUndefined();
  });
});

describe("warning logging", () => {
  it("logs warning when falling back to schema value for label", () => {
    const mockT = vi.fn(
      (_key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? "",
    ) as unknown as TFunction;

    resolveSchemaFieldLabel(mockT, "unknown.field", "Fallback Label");

    expect(warnSpy).toHaveBeenCalledWith(
      '[i18n] Missing translation for key "SCHEMA$UNKNOWN$FIELD$LABEL", falling back to: "Fallback Label"',
    );
  });

  it("logs warning when falling back to schema value for description", () => {
    const mockT = vi.fn(
      (_key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? "",
    ) as unknown as TFunction;

    resolveSchemaFieldDescription(
      mockT,
      "unknown.field",
      "Fallback Description",
    );

    expect(warnSpy).toHaveBeenCalledWith(
      '[i18n] Missing translation for key "SCHEMA$UNKNOWN$FIELD$DESCRIPTION", falling back to: "Fallback Description"',
    );
  });

  it("logs warning when falling back to schema value for choice label", () => {
    const mockT = vi.fn(
      (_key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? "",
    ) as unknown as TFunction;

    resolveSchemaChoiceLabel(mockT, "some.field", "option1", "Option 1");

    expect(warnSpy).toHaveBeenCalledWith(
      '[i18n] Missing translation for key "SCHEMA$SOME$FIELD$CHOICE$OPTION1", falling back to: "Option 1"',
    );
  });

  it("does not log warning when translation is found", () => {
    const mockT = vi.fn((key: string) => {
      if (key === "SCHEMA$LLM$API_KEY$LABEL") return "Translated API Key";
      return "";
    }) as unknown as TFunction;

    resolveSchemaFieldLabel(mockT, "llm.api_key", "API Key");

    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("does not log warning when description is null/undefined", () => {
    const mockT = vi.fn(
      (_key: string, options?: { defaultValue: string }) =>
        options?.defaultValue ?? "",
    ) as unknown as TFunction;

    resolveSchemaFieldDescription(mockT, "some.field", null);
    resolveSchemaFieldDescription(mockT, "some.field", undefined);

    expect(warnSpy).not.toHaveBeenCalled();
  });
});
