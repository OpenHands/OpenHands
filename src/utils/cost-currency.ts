export const COST_CURRENCIES = [
  "USD",
  "EUR",
  "GBP",
  "JPY",
  "CNY",
  "KRW",
  "CAD",
  "AUD",
  "CHF",
  "SEK",
  "NOK",
  "INR",
  "BRL",
  "MXN",
  "SGD",
  "HKD",
  "PLN",
] as const;

export type CostCurrency = (typeof COST_CURRENCIES)[number];

export const COST_CURRENCY_STORAGE_KEY = "openhands-cost-currency";
export const FRANKFURTER_API_BASE = "https://api.frankfurter.app";

export interface FormatCostAmountOptions {
  detailed?: boolean;
  locale?: string;
}

export function isCostCurrency(value: string): value is CostCurrency {
  return (COST_CURRENCIES as readonly string[]).includes(value);
}

export function usdRateDayKey(at?: string | Date | number | null): string {
  const now = new Date();
  const parsed =
    at == null || at === "" ? now : at instanceof Date ? at : new Date(at);
  const date = Number.isNaN(parsed.getTime()) ? now : parsed;
  const clamped = date.getTime() > now.getTime() ? now : date;
  return clamped.toISOString().slice(0, 10);
}

export function usdRateCacheKey(currency: CostCurrency, day: string): string {
  return `${day}:${currency}`;
}

export function formatCostAmount(
  amount: number | null | undefined,
  currency: CostCurrency,
  options?: FormatCostAmountOptions,
): string {
  const locale = options?.locale ?? "en-US";
  const resolved = new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
  }).resolvedOptions();
  const zeroDecimal = (resolved.minimumFractionDigits ?? 0) === 0;
  const digits = zeroDecimal
    ? { minimumFractionDigits: 0, maximumFractionDigits: 0 }
    : options?.detailed
      ? { minimumFractionDigits: 4, maximumFractionDigits: 4 }
      : {
          minimumFractionDigits: resolved.minimumFractionDigits ?? 2,
          maximumFractionDigits: resolved.maximumFractionDigits ?? 2,
        };

  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
    ...digits,
  }).format(Number(amount ?? 0));
}

export async function fetchUsdToRate(
  currency: CostCurrency,
  day: string,
): Promise<number | null> {
  if (currency === "USD") return 1;
  try {
    const today = usdRateDayKey(null);
    const path = day === today ? "latest" : day;
    const url = `${FRANKFURTER_API_BASE}/${path}?from=USD&to=${encodeURIComponent(currency)}`;
    const response = await fetch(url);
    if (!response.ok) return null;
    const body = (await response.json()) as { rates?: Record<string, number> };
    const rate = body.rates?.[currency];
    return typeof rate === "number" && Number.isFinite(rate) && rate > 0
      ? rate
      : null;
  } catch {
    return null;
  }
}
