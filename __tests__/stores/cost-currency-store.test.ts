import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COST_CURRENCY_STORAGE_KEY } from "#/utils/cost-currency";
import {
  convertUsd,
  formatUsd,
  useCostCurrencyStore,
} from "#/stores/cost-currency-store";

function gbp(amount: number, detailed = false): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "GBP",
    minimumFractionDigits: detailed ? 4 : 2,
    maximumFractionDigits: detailed ? 4 : 2,
  }).format(amount);
}

describe("cost currency store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useCostCurrencyStore.setState({
      currency: "USD",
      rates: {},
      latestRates: {},
      rateRevision: 0,
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("leaves USD amounts unchanged", () => {
    expect(formatUsd(1.25)).toBe("$1.25");
    expect(convertUsd(10, "2024-01-15T00:00:00Z")).toBe(10);
  });

  it("converts each cost at that day's rate instead of the latest rate", () => {
    useCostCurrencyStore.setState({
      currency: "GBP",
      latestRates: { GBP: 0.8 },
      rates: {
        "2024-01-15:GBP": 0.79,
        "2024-06-01:GBP": 0.75,
      },
      rateRevision: 1,
    });

    expect(formatUsd(1, "2024-01-15T12:00:00Z")).toBe(gbp(0.79));
    expect(formatUsd(1, "2024-06-01T12:00:00Z")).toBe(gbp(0.75));
    expect(formatUsd(1)).toBe(gbp(0.8));

    const convertedSum =
      convertUsd(10, "2024-01-15T00:00:00Z") +
      convertUsd(10, "2024-06-01T00:00:00Z");
    expect(convertedSum).toBeCloseTo(15.4);
    expect(convertUsd(20)).toBeCloseTo(16);
  });

  it("falls back to the latest rate when a historical day is missing", () => {
    useCostCurrencyStore.setState({
      currency: "GBP",
      latestRates: { GBP: 0.8 },
      rates: {},
      rateRevision: 1,
    });

    expect(convertUsd(10, "2020-01-02T00:00:00Z")).toBeCloseTo(8);
  });

  it("persists the selected currency", () => {
    useCostCurrencyStore.getState().setCurrency("GBP");

    const persisted = JSON.parse(
      window.localStorage.getItem(COST_CURRENCY_STORAGE_KEY) ?? "{}",
    ) as { state?: { currency?: string } };
    expect(persisted.state?.currency).toBe("GBP");
    expect(useCostCurrencyStore.getState().currency).toBe("GBP");
  });

  it("fetches a historical USD rate from Frankfurter", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ rates: { GBP: 0.79 }, date: "2024-01-15" }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    useCostCurrencyStore.getState().setCurrency("GBP");
    const rate = await useCostCurrencyStore
      .getState()
      .ensureUsdRate("GBP", "2024-01-15T00:00:00Z");

    expect(rate).toBe(0.79);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const requested = String(fetchMock.mock.calls.at(0)?.at(0));
    expect(requested).toContain("api.frankfurter.app/2024-01-15");
    expect(requested).toContain("from=USD");
    expect(requested).toContain("to=GBP");
    expect(useCostCurrencyStore.getState().rates["2024-01-15:GBP"]).toBe(0.79);
  });

  it("does not throw when the FX request fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("network down");
      }),
    );

    useCostCurrencyStore.getState().setCurrency("GBP");
    await expect(
      useCostCurrencyStore.getState().ensureUsdRate("GBP", "2024-01-15"),
    ).resolves.toBe(1);
  });
});
