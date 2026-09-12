import { describe, expect, it } from "vitest";
import {
  COST_CURRENCIES,
  formatCostAmount,
  isCostCurrency,
  usdRateCacheKey,
  usdRateDayKey,
} from "#/utils/cost-currency";

describe("cost currency helpers", () => {
  it("accepts only the supported currency codes", () => {
    expect(isCostCurrency("USD")).toBe(true);
    expect(isCostCurrency("GBP")).toBe(true);
    expect(isCostCurrency("usd")).toBe(false);
    expect(isCostCurrency("BTC")).toBe(false);
    expect(COST_CURRENCIES).toContain("EUR");
  });

  it("keys FX lookups by UTC calendar day", () => {
    expect(usdRateDayKey("2024-01-15T23:30:00Z")).toBe("2024-01-15");
    expect(usdRateCacheKey("GBP", "2024-01-15")).toBe("2024-01-15:GBP");
  });

  it("clamps future timestamps to today", () => {
    const future = new Date();
    future.setUTCDate(future.getUTCDate() + 10);
    expect(usdRateDayKey(future.toISOString())).toBe(usdRateDayKey(null));
  });

  it("formats amounts in the selected currency", () => {
    expect(formatCostAmount(1.25, "USD")).toBe("$1.25");
    expect(formatCostAmount(0.79, "GBP")).toBe(
      new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "GBP",
      }).format(0.79),
    );
    expect(formatCostAmount(0.4213, "USD", { detailed: true })).toBe("$0.4213");
    expect(formatCostAmount(0, "USD", { detailed: true })).toBe("$0.0000");
  });
});
