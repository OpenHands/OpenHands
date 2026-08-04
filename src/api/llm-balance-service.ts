import { getAgentServerClientOptions } from "./agent-server-client-options";
import { LLM_BALANCE_PATH } from "#/constants/llm-balance";

/**
 * Credit balance reported by the agent server for the active LLM provider
 * (currently only OpenRouter). Amounts are in USD credits.
 */
export interface LLMBalance {
  provider: string;
  /** Credit cap configured for the key, or null when uncapped. */
  limit: number | null;
  /** Credits remaining under `limit`, or null when uncapped. */
  limitRemaining: number | null;
  /** Total credits consumed by the key. */
  usage: number;
  usageDaily: number | null;
  usageWeekly: number | null;
  usageMonthly: number | null;
  isFreeTier: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeBalance(raw: unknown): LLMBalance | null {
  if (!isRecord(raw) || typeof raw.provider !== "string") return null;

  return {
    provider: raw.provider,
    limit: numberOrNull(raw.limit),
    limitRemaining: numberOrNull(raw.limit_remaining),
    usage: numberOrNull(raw.usage) ?? 0,
    usageDaily: numberOrNull(raw.usage_daily),
    usageWeekly: numberOrNull(raw.usage_weekly),
    usageMonthly: numberOrNull(raw.usage_monthly),
    isFreeTier: raw.is_free_tier === true,
  };
}

class LLMBalanceService {
  /**
   * Fetch the provider balance from the agent server.
   *
   * Returns `null` when the server answers 404 (older servers without the
   * endpoint, or a provider that doesn't support balance reporting) or when
   * the response doesn't match the expected shape — callers hide the balance
   * UI in that case. Other HTTP failures throw.
   */
  static async getBalance(): Promise<LLMBalance | null> {
    const { host, apiKey } = getAgentServerClientOptions();
    const headers = new Headers({ Accept: "application/json" });
    if (apiKey) {
      headers.set("X-Session-API-Key", apiKey);
    }

    const response = await fetch(`${host}${LLM_BALANCE_PATH}`, { headers });
    if (response.status === 404) return null;
    if (!response.ok) {
      throw new Error(`Balance request failed with ${response.status}`);
    }
    return normalizeBalance(await response.json());
  }
}

export default LLMBalanceService;
