import { describe, expect, it } from "vitest";
import i18next from "i18next";
import { translationResources } from "#/i18n/resources";

const FORMAT = translationResources.en.CONVERSATION$BUDGET_USAGE_FORMAT;

// The exact parameters BudgetUsageText passes to t() (see
// src/components/features/conversation-panel/budget-usage-text.tsx).
const CONSUMER_PARAMS = {
  currentCost: "$0.1234",
  maxBudget: "$5.0000",
  usagePercentage: "2.47",
  used: translationResources.en.CONVERSATION$USED,
};

describe("CONVERSATION$BUDGET_USAGE_FORMAT", () => {
  it("uses i18next placeholder syntax in the en locale string", () => {
    // Regression guard for #16873: the key must use {{param}} placeholders so
    // i18next substitutes values instead of printing the literal template.
    expect(FORMAT).toBe(
      "{{currentCost}} / {{maxBudget}} ({{usagePercentage}}% {{used}})",
    );
    expect(FORMAT).not.toContain("${");
  });

  it("interpolates every named parameter the consumer passes", () => {
    for (const param of Object.keys(CONSUMER_PARAMS)) {
      expect(FORMAT).toContain(`{{${param}}}`);
    }
  });

  it("renders a fully substituted usage line with i18next", () => {
    // Drive a real i18next instance with the actual en resource to prove the
    // template interpolates end-to-end (the app's interpolation config matches:
    // default delimiters, escapeValue false).
    const instance = i18next.createInstance();
    instance.init({
      lng: "en",
      resources: { en: { openhands: translationResources.en } },
      ns: ["openhands"],
      defaultNS: "openhands",
      interpolation: { escapeValue: false },
    });

    const rendered = instance.t(
      "CONVERSATION$BUDGET_USAGE_FORMAT",
      CONSUMER_PARAMS,
    ) as string;

    expect(rendered).toBe("$0.1234 / $5.0000 (2.47% used)");
    expect(rendered).not.toContain("${");
    expect(rendered).not.toContain("{{");
  });
});
