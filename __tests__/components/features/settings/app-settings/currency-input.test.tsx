import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CurrencyInput } from "#/components/features/settings/app-settings/currency-input";
import { useCostCurrencyStore } from "#/stores/cost-currency-store";

describe("CurrencyInput", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useCostCurrencyStore.setState({
      currency: "USD",
      rates: {},
      latestRates: {},
      rateRevision: 0,
    });
  });

  it("renders the cost currency control", () => {
    render(<CurrencyInput />);

    expect(screen.getByTestId("cost-currency-input")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$COST_CURRENCY")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$COST_CURRENCY_HINT")).toBeInTheDocument();
  });

  it("updates the store when a currency is selected", async () => {
    const user = userEvent.setup();
    render(<CurrencyInput />);

    await user.click(screen.getByTestId("cost-currency-input"));
    await user.click(await screen.findByText("GBP"));

    expect(useCostCurrencyStore.getState().currency).toBe("GBP");
  });
});
