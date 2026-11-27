import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import {
  LLM_BALANCE_PATH,
  LLM_BALANCE_TIMEOUT_MS,
} from "#/constants/llm-balance";

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

/** `AbortSignal.timeout` rejects with a `TimeoutError`; a manual abort uses `AbortError`. */
function isAbortLike(error: unknown): boolean {
  return (
    isRecord(error) &&
    (error.name === "TimeoutError" || error.name === "AbortError")
  );
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
   * UI in that case. Other HTTP failures throw, as does a request that outlives
   * `LLM_BALANCE_TIMEOUT_MS`. A timeout stays an error rather than collapsing
   * into `null`: `null` means "this server has no balance to report" and is
   * cached under `staleTime: Infinity`, so treating a transient stall as an
   * absent endpoint would hide the card for the rest of the conversation.
   */
  static async getBalance(): Promise<LLMBalance | null> {
    const { host, apiKey } = getAgentServerClientOptions();
    const client = new AgentServerClient({
      host,
      apiKey,
      timeout: LLM_BALANCE_TIMEOUT_MS,
    });
    try {
      const data = await client.get<unknown>(LLM_BALANCE_PATH, {
        acceptableStatusCodes: new Set([200]),
        responseType: "json",
        timeoutSeconds: Math.floor(LLM_BALANCE_TIMEOUT_MS / 1000),
      });
      return normalizeBalance(data);
    } catch (error) {
      // Older servers without balance support answer 404 — hide the UI.
      if (
        error instanceof Error &&
        "status" in error &&
        (error as { status: number }).status === 404
      ) {
        return null;
      }
      if (isAbortLike(error)) {
        throw new Error(
          `Balance request timed out after ${LLM_BALANCE_TIMEOUT_MS}ms`,
        );
      }
      const status =
        error instanceof Error &&
        "status" in error &&
        (error as { status: number }).status;
      throw new Error(
        status
          ? `Balance request failed with ${status}`
          : `Balance request failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.close();
    }
  }
}

export default LLMBalanceService;
