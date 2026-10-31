import React from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { createInstance } from "i18next";
import { I18nextProvider } from "react-i18next";
import { renderWithProviders } from "test-utils";
import { BudgetDisplay } from "#/components/features/conversation-panel/budget-display";
import { translationResources } from "#/i18n/resources";

vi.unmock("react-i18next");

const i18n = createInstance();

beforeAll(async () => {
  await i18n.init({
    lng: "en",
    resources: { en: { openhands: translationResources.en } },
    ns: ["openhands"],
    defaultNS: "openhands",
    interpolation: { escapeValue: false },
  });
});

function renderBudget(cost: number | null, maxBudget: number | null) {
  return renderWithProviders(
    <I18nextProvider i18n={i18n}>
      <BudgetDisplay cost={cost} maxBudget={maxBudget} />
    </I18nextProvider>,
  );
}

describe("BudgetDisplay", () => {
  it("renders nothing when cost is unavailable", () => {
    const { container } = renderBudget(null, 5);
    expect(container).toBeEmptyDOMElement();
  });

  it.each([null, 0, -1])("renders nothing for cap %s", (cap) => {
    const { container } = renderBudget(1.23, cap);
    expect(container).toBeEmptyDOMElement();
  });

  it.each([
    [1.23, "$1.2300 / $5.0000 (24.60% used)", "24.6%"],
    [6.25, "$6.2500 / $5.0000 (125.00% used)", "100%"],
  ] as const)(
    "shows actual spend %s and bounds only the meter width",
    (cost, text, width) => {
      const { container } = renderBudget(cost, 5);

      expect(screen.getByText(text)).toBeInTheDocument();
      expect(container.querySelector("[style]")).toHaveStyle({ width });
    },
  );
});
