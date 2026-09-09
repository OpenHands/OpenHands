import React from "react";
import { useTranslation } from "react-i18next";
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import {
  COST_CURRENCY_STORAGE_KEY,
  fetchUsdToRate,
  formatCostAmount,
  isCostCurrency,
  usdRateCacheKey,
  usdRateDayKey,
  type CostCurrency,
  type FormatCostAmountOptions,
} from "#/utils/cost-currency";

interface CostCurrencyState {
  currency: CostCurrency;
  rates: Record<string, number>;
  latestRates: Partial<Record<CostCurrency, number>>;
  rateRevision: number;
}

interface CostCurrencyActions {
  setCurrency: (currency: string) => void;
  ensureUsdRate: (
    currency?: CostCurrency,
    at?: string | Date | number | null,
  ) => Promise<number>;
}

type CostCurrencyStore = CostCurrencyState & CostCurrencyActions;

const inflightUsdRates = new Map<string, Promise<number | null>>();

const initialState: CostCurrencyState = {
  currency: "USD",
  rates: {},
  latestRates: {},
  rateRevision: 0,
};

function rateFromState(
  state: CostCurrencyState,
  currency: CostCurrency,
  day: string,
): number {
  if (currency === "USD") return 1;
  return (
    state.rates[usdRateCacheKey(currency, day)] ??
    state.latestRates[currency] ??
    1
  );
}

export const useCostCurrencyStore = create<CostCurrencyStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      setCurrency: (currency) => {
        if (!isCostCurrency(currency) || currency === get().currency) return;
        set((state) => ({
          currency,
          rateRevision: state.rateRevision + 1,
        }));
      },

      ensureUsdRate: async (currency = get().currency, at) => {
        if (currency === "USD") return 1;
        const day = usdRateDayKey(at);
        const key = usdRateCacheKey(currency, day);
        const cached = get().rates[key];
        if (cached != null) return cached;
        const today = usdRateDayKey(null);
        if (day === today) {
          const latest = get().latestRates[currency];
          if (latest != null) return latest;
        }

        const pending = inflightUsdRates.get(key);
        if (pending) {
          const resolved = await pending;
          return resolved ?? rateFromState(get(), currency, day);
        }

        const request = fetchUsdToRate(currency, day).finally(() => {
          inflightUsdRates.delete(key);
        });
        inflightUsdRates.set(key, request);
        const rate = await request;
        if (rate == null) return rateFromState(get(), currency, day);

        set((state) => ({
          rates: { ...state.rates, [key]: rate },
          latestRates:
            day === usdRateDayKey(null)
              ? { ...state.latestRates, [currency]: rate }
              : state.latestRates,
          rateRevision: state.rateRevision + 1,
        }));
        return rate;
      },
    }),
    {
      name: COST_CURRENCY_STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state): Pick<CostCurrencyState, "currency"> => ({
        currency: state.currency,
      }),
      merge: (persisted, current) => {
        const stored = persisted as
          | Pick<CostCurrencyState, "currency">
          | undefined;
        const currency =
          stored && isCostCurrency(stored.currency)
            ? stored.currency
            : current.currency;
        return { ...current, currency };
      },
    },
  ),
);

export function convertUsd(
  amountUsd: number | null | undefined,
  at?: string | Date | number | null,
): number {
  const state = useCostCurrencyStore.getState();
  const { currency } = state;
  const amount = Number(amountUsd) || 0;
  if (currency === "USD") return amount;
  const day = usdRateDayKey(at);
  void state.ensureUsdRate(currency, at);
  return amount * rateFromState(state, currency, day);
}

export function formatUsd(
  amount: number | null | undefined,
  at?: string | Date | number | null,
  options?: FormatCostAmountOptions,
): string {
  const { currency } = useCostCurrencyStore.getState();
  return formatCostAmount(convertUsd(amount, at), currency, options);
}

export function useFormatCost() {
  const currency = useCostCurrencyStore((state) => state.currency);
  const revision = useCostCurrencyStore((state) => state.rateRevision);
  const { i18n } = useTranslation();

  React.useEffect(() => {
    if (currency === "USD") return;
    void useCostCurrencyStore.getState().ensureUsdRate(currency);
  }, [currency]);

  return React.useCallback(
    (
      amount: number | null | undefined,
      at?: string | Date | number | null,
      options?: Pick<FormatCostAmountOptions, "detailed">,
    ) =>
      formatUsd(amount, at, {
        ...options,
        locale: i18n.language || "en-US",
      }),
    [currency, revision, i18n.language],
  );
}

export function useConvertUsd() {
  const currency = useCostCurrencyStore((state) => state.currency);
  const revision = useCostCurrencyStore((state) => state.rateRevision);

  return React.useCallback(
    (amount: number | null | undefined, at?: string | Date | number | null) =>
      convertUsd(amount, at),
    [currency, revision],
  );
}

export function useFormatCostAmount() {
  const currency = useCostCurrencyStore((state) => state.currency);
  const { i18n } = useTranslation();

  return React.useCallback(
    (
      amount: number | null | undefined,
      options?: Pick<FormatCostAmountOptions, "detailed">,
    ) =>
      formatCostAmount(amount, currency, {
        ...options,
        locale: i18n.language || "en-US",
      }),
    [currency, i18n.language],
  );
}
