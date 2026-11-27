import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  LLM_BALANCE_PATH,
  LLM_BALANCE_TIMEOUT_MS,
} from "#/constants/llm-balance";
import LLMBalanceService from "./llm-balance-service";
import { AgentServerClient } from "@openhands/typescript-client/clients";

const getMock = vi.hoisted(() => vi.fn());

vi.mock("@openhands/typescript-client/clients", () => ({
  AgentServerClient: vi.fn(function AgentServerClientMock() {
    return { get: getMock, close: vi.fn() };
  }),
}));

const localBackend: Backend = {
  id: "local-test",
  name: "Local test backend",
  host: "http://localhost:3000",
  apiKey: "test-session-key",
  kind: "local",
};

const balancePayload = {
  provider: "openrouter",
  limit: 20,
  limit_remaining: 14.96,
  usage: 5.04,
  usage_daily: 1.2,
  usage_weekly: 3.4,
  usage_monthly: 5.04,
  is_free_tier: false,
};

function httpError(status: number): Error & { status: number } {
  return Object.assign(new Error(`HTTP ${status}`), { status });
}

function mockedClient(): AgentServerClient {
  return vi.mocked(AgentServerClient).mock.results[0]
    ?.value as AgentServerClient;
}

describe("LLMBalanceService.getBalance", () => {
  beforeEach(() => {
    getMock.mockReset();
    vi.mocked(AgentServerClient).mockClear();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
  });

  it("normalizes a successful balance response via the typed client", async () => {
    getMock.mockResolvedValue(balancePayload);

    const balance = await LLMBalanceService.getBalance();

    expect(balance).toEqual({
      provider: "openrouter",
      limit: 20,
      limitRemaining: 14.96,
      usage: 5.04,
      usageDaily: 1.2,
      usageWeekly: 3.4,
      usageMonthly: 5.04,
      isFreeTier: false,
    });

    expect(vi.mocked(AgentServerClient)).toHaveBeenCalledWith({
      host: "http://localhost:3000",
      apiKey: "test-session-key",
      timeout: LLM_BALANCE_TIMEOUT_MS,
    });
    expect(getMock).toHaveBeenCalledWith(LLM_BALANCE_PATH, {
      acceptableStatusCodes: new Set([200]),
      responseType: "json",
      timeoutSeconds: Math.floor(LLM_BALANCE_TIMEOUT_MS / 1000),
    });
    expect(mockedClient().close).toHaveBeenCalled();
  });

  it("treats null caps as an uncapped key", async () => {
    getMock.mockResolvedValue({
      ...balancePayload,
      limit: null,
      limit_remaining: null,
    });

    const balance = await LLMBalanceService.getBalance();

    expect(balance?.limit).toBeNull();
    expect(balance?.limitRemaining).toBeNull();
    expect(balance?.usage).toBe(5.04);
  });

  it("returns null when the endpoint is missing (404)", async () => {
    getMock.mockRejectedValue(httpError(404));

    await expect(LLMBalanceService.getBalance()).resolves.toBeNull();
  });

  it("returns null for a malformed response body", async () => {
    getMock.mockResolvedValue({ unexpected: true });

    await expect(LLMBalanceService.getBalance()).resolves.toBeNull();
  });

  it("throws on non-404 HTTP failures", async () => {
    getMock.mockRejectedValue(httpError(500));

    await expect(LLMBalanceService.getBalance()).rejects.toThrow(
      "Balance request failed with 500",
    );
  });

  it("throws a named error when the request times out", async () => {
    getMock.mockRejectedValue(
      new DOMException("The operation was aborted.", "TimeoutError"),
    );

    await expect(LLMBalanceService.getBalance()).rejects.toThrow(
      "Balance request timed out",
    );
  });
});
