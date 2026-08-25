import { describe, expect, it } from "vitest";
import fs from "fs";
import path from "path";
import { createInstance } from "i18next";

// Guard against malformed interpolation placeholders in translation.json.
//
// i18next only substitutes `{{param}}` (double braces) by default, and
// src/i18n/index.ts does not override the interpolation delimiters. A value
// written with JS-template syntax (`${param}`) or single braces (`{param}`)
// is therefore rendered literally to the user.
//
// Regression test for #16873, where CONVERSATION$BUDGET_USAGE_FORMAT rendered
// the literal string "${currentCost} / ${maxBudget} ({usagePercentage}% {used})"
// in all locales.

type TranslationMap = Record<string, Record<string, string>>;

const translationJson: TranslationMap = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, "../../src/i18n/translation.json"),
    "utf-8",
  ),
);

// Keys allowed to contain brace text that is NOT an i18next placeholder.
// Add a key here only with a comment explaining why it is safe.
const SINGLE_BRACE_ALLOWLIST = new Set<string>([
  // Literal JSON example shown in the MCP settings editor — the braces are
  // real JSON syntax, not placeholders.
  "SETTINGS$MCP_DEFAULT_CONFIG",
  // Pre-existing, currently unreferenced keys whose CJK values embed
  // `{minutes}`-style tokens while the English value has no placeholder at
  // all. Grandfathered so this guard can land; fixing or removing them is
  // tracked separately from #16873.
  "TIME$MINUTES_AGO",
  "TIME$HOURS_AGO",
  "TIME$DAYS_AGO",
]);

// A single-brace token that looks like an interpolation placeholder:
// `{ident}` not part of `{{ident}}`.
const SINGLE_BRACE_PLACEHOLDER = /(?<!\{)\{\s*[A-Za-z_][A-Za-z0-9_]*\s*\}(?!\})/;

describe("translation.json interpolation placeholders", () => {
  it("contains no JS-template `${...}` placeholders", () => {
    const offenders: string[] = [];

    Object.entries(translationJson).forEach(([key, locales]) => {
      Object.entries(locales).forEach(([lang, value]) => {
        if (typeof value === "string" && value.includes("${")) {
          offenders.push(`${key} (${lang}): ${value}`);
        }
      });
    });

    expect(
      offenders,
      `i18next does not interpolate \${...}; use {{param}} instead:\n${offenders.join("\n")}`,
    ).toEqual([]);
  });

  it("contains no single-brace `{param}` placeholders", () => {
    const offenders: string[] = [];

    Object.entries(translationJson).forEach(([key, locales]) => {
      if (SINGLE_BRACE_ALLOWLIST.has(key)) return;

      Object.entries(locales).forEach(([lang, value]) => {
        if (typeof value === "string" && SINGLE_BRACE_PLACEHOLDER.test(value)) {
          offenders.push(`${key} (${lang}): ${value}`);
        }
      });
    });

    expect(
      offenders,
      `i18next does not interpolate single-brace {param}; use {{param}} instead:\n${offenders.join("\n")}`,
    ).toEqual([]);
  });
});

describe("CONVERSATION$BUDGET_USAGE_FORMAT", () => {
  const key = "CONVERSATION$BUDGET_USAGE_FORMAT";
  const params = ["currentCost", "maxBudget", "usagePercentage", "used"];

  it("declares all four {{...}} params in every locale", () => {
    const locales = translationJson[key];
    expect(locales).toBeDefined();

    Object.entries(locales).forEach(([lang, value]) => {
      params.forEach((param) => {
        expect(value, `${key} (${lang}) is missing {{${param}}}`).toContain(
          `{{${param}}}`,
        );
      });
    });
  });

  it("interpolates real values with i18next's default delimiters", async () => {
    // Same shape scripts/make-i18n-translations.cjs produces and the same
    // interpolation config src/i18n/index.ts uses.
    const i18n = createInstance();
    await i18n.init({
      lng: "en",
      ns: ["openhands"],
      defaultNS: "openhands",
      resources: {
        en: {
          openhands: {
            [key]: translationJson[key].en,
            CONVERSATION$USED: translationJson.CONVERSATION$USED.en,
          },
        },
      },
      interpolation: { escapeValue: false },
    });

    // Mirrors src/components/features/conversation-panel/budget-usage-text.tsx
    const rendered = i18n.t(key, {
      currentCost: "$0.1234",
      maxBudget: "$5.0000",
      usagePercentage: "2.47",
      used: i18n.t("CONVERSATION$USED"),
    });

    expect(rendered).toBe("$0.1234 / $5.0000 (2.47% used)");
    expect(rendered).not.toContain("{");
  });
});
